package pipeline

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/go-github/v57/github"
	"github.com/sirupsen/logrus"

	"github.com/twin2ai/studio/internal/config"
	githubclient "github.com/twin2ai/studio/internal/github"
	"github.com/twin2ai/studio/internal/multiprovider"
)

type BatchPipeline struct {
	config           *config.Config
	github           *githubclient.Client
	multiGenerator   *multiprovider.Generator
	logger           *logrus.Logger
	processedNames   map[string]bool
	processedAliases map[string]string // Maps tracking keys to full names
	force            bool
}

func NewBatchPipeline(cfg *config.Config, githubClient *githubclient.Client, multiGen *multiprovider.Generator, logger *logrus.Logger, force bool) (*BatchPipeline, error) {
	bp := &BatchPipeline{
		config:           cfg,
		github:           githubClient,
		multiGenerator:   multiGen,
		logger:           logger,
		processedNames:   make(map[string]bool),
		processedAliases: make(map[string]string),
		force:            force,
	}

	// Load processed names from file
	if err := bp.loadProcessedNames(); err != nil {
		logger.Warnf("Failed to load processed names: %v", err)
	}

	return bp, nil
}

func (bp *BatchPipeline) ProcessFile(ctx context.Context, filePath string) error {
	// Open the file
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// Read names from file
	var personaNames []*PersonaName
	scanner := bufio.NewScanner(file)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Parse the name
		personaName, err := ParsePersonaName(line)
		if err != nil {
			bp.logger.Warnf("Line %d: Failed to parse name '%s': %v", lineNum, line, err)
			continue
		}

		// Validate parsed name
		if err := bp.validatePersonaName(personaName); err != nil {
			bp.logger.Warnf("Line %d: Invalid name '%s': %v", lineNum, line, err)
			continue
		}

		personaNames = append(personaNames, personaName)
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %w", err)
	}

	bp.logger.Infof("Found %d valid names to process", len(personaNames))

	// Process each name
	successCount := 0
	skipCount := 0
	errorCount := 0

	for i, personaName := range personaNames {
		bp.logger.Infof("[%d/%d] Processing: %s", i+1, len(personaNames), personaName.FullName)

		// Check if already processed (unless force flag is set)
		trackingKey := personaName.GetTrackingKey()
		if !bp.force && bp.processedNames[trackingKey] {
			bp.logger.Infof("  → Already processed in previous batch, skipping")
			skipCount++
			continue
		}

		// Check if persona already exists in repo (unless force flag is set)
		if !bp.force {
			exists, err := bp.checkPersonaExists(ctx, personaName)
			if err != nil {
				bp.logger.Errorf("  → Error checking if persona exists: %v", err)
				errorCount++
				continue
			}
			if exists {
				bp.logger.Infof("  → Persona already exists in repository, skipping")
				skipCount++
				// Still mark as processed to avoid future checks
				bp.processedNames[trackingKey] = true
				bp.processedAliases[trackingKey] = personaName.FullName
				if err := bp.saveProcessedName(personaName); err != nil {
					bp.logger.Warnf("  → Failed to save processed name: %v", err)
				}
				continue
			}
		}

		// Generate persona
		if err := bp.generatePersona(ctx, personaName); err != nil {
			bp.logger.Errorf("  → Failed to generate persona: %v", err)
			errorCount++
			continue
		}

		// Mark as processed
		bp.processedNames[trackingKey] = true
		bp.processedAliases[trackingKey] = personaName.FullName
		if err := bp.saveProcessedName(personaName); err != nil {
			bp.logger.Warnf("  → Failed to save processed name: %v", err)
		}

		successCount++
		bp.logger.Infof("  → Successfully generated persona")

		// Add a small delay between requests to avoid rate limiting
		if i < len(personaNames)-1 {
			time.Sleep(2 * time.Second)
		}
	}

	// Summary
	bp.logger.Info("=== Batch Processing Complete ===")
	bp.logger.Infof("Total names: %d", len(personaNames))
	bp.logger.Infof("Successful: %d", successCount)
	bp.logger.Infof("Skipped: %d", skipCount)
	bp.logger.Infof("Errors: %d", errorCount)

	return nil
}

func (bp *BatchPipeline) validatePersonaName(pn *PersonaName) error {
	// Validate primary name
	if len(pn.PrimaryName) < 2 {
		return fmt.Errorf("primary name too short (minimum 2 characters)")
	}
	if len(pn.PrimaryName) > 100 {
		return fmt.Errorf("primary name too long (maximum 100 characters)")
	}

	// Validate real name if different
	if pn.HasAlias() {
		if len(pn.RealName) < 2 {
			return fmt.Errorf("real name too short (minimum 2 characters)")
		}
		if len(pn.RealName) > 100 {
			return fmt.Errorf("real name too long (maximum 100 characters)")
		}
	}

	return nil
}

func (bp *BatchPipeline) checkPersonaExists(ctx context.Context, personaName *PersonaName) (bool, error) {
	// Check all variations of the name
	variations := personaName.GetSearchVariations()

	for _, variation := range variations {
		// Search for existing persona files in the personas repository
		// We'll check for directories that might match any variation
		sanitizedName := bp.sanitizeForPath(variation)

		// Try direct path check first
		path := fmt.Sprintf("personas/%s/synthesized.md", sanitizedName)
		_, _, _, err := bp.github.GetClient().Repositories.GetContents(
			ctx,
			bp.config.GitHub.PersonasOwner,
			bp.config.GitHub.PersonasRepo,
			path,
			&github.RepositoryContentGetOptions{},
		)

		if err == nil {
			// Found a match
			bp.logger.Debugf("Found existing persona at: %s", path)
			return true, nil
		}

		// If not a 404, it's a real error
		if !strings.Contains(err.Error(), "404") {
			bp.logger.Warnf("Error checking path %s: %v", path, err)
		}
	}

	// Also try searching by content for any mentions of the names
	// This helps catch cases where the directory name might be different
	if personaName.HasAlias() {
		query := fmt.Sprintf("repo:%s/%s path:personas \"%s\" OR \"%s\"",
			bp.config.GitHub.PersonasOwner,
			bp.config.GitHub.PersonasRepo,
			personaName.PrimaryName,
			personaName.RealName)

		searchOpts := &github.SearchOptions{
			ListOptions: github.ListOptions{
				PerPage: 5,
			},
		}

		results, _, err := bp.github.GetClient().Search.Code(ctx, query, searchOpts)
		if err != nil {
			bp.logger.Debugf("Search API failed: %v", err)
			// Don't fail on search errors, continue with generation
			return false, nil
		}

		if results.GetTotal() > 0 {
			// Found potential matches, check if they're actual personas
			for _, result := range results.CodeResults {
				if result.Path != nil && strings.Contains(*result.Path, "/synthesized.md") {
					bp.logger.Debugf("Found potential match in: %s", *result.Path)
					return true, nil
				}
			}
		}
	}

	return false, nil
}

func (bp *BatchPipeline) generatePersona(ctx context.Context, personaName *PersonaName) error {
	// Create a description that includes both names if available
	description := fmt.Sprintf("Batch processing request for persona: %s", personaName.GetPromptDescription())
	if personaName.HasAlias() {
		description += fmt.Sprintf("\n\nThis persona is for %s, who is also known as %s.",
			personaName.PrimaryName, personaName.RealName)
	}
	description += "\n\nGenerated via studio batch command."

	// Create a mock issue for the persona generation
	// Issue number 0 indicates batch processing
	issue := &github.Issue{
		Number: github.Int(0),
		Title:  github.String(personaName.FullName),
		Body:   github.String(description),
	}

	// Generate persona using the multi-provider approach
	_, files, err := bp.multiGenerator.ProcessIssueWithStructure(ctx, issue)
	if err != nil {
		return fmt.Errorf("failed to generate persona: %w", err)
	}

	// Create PR for the persona
	// Use the full name for the PR title
	pr, err := bp.github.CreateStructuredPersonaPR(ctx,
		0, // Use 0 for batch processing
		personaName.FullName,
		*files)
	if err != nil {
		// Check if it's a duplicate PR error
		if strings.Contains(err.Error(), "A pull request already exists") {
			bp.logger.Warnf("  → PR already exists for this persona")
			return nil // Don't treat as error
		}
		return fmt.Errorf("failed to create PR: %w", err)
	}

	bp.logger.Infof("  → Created PR #%d", *pr.Number)
	return nil
}

func (bp *BatchPipeline) sanitizeForPath(name string) string {
	// Convert name to a valid path component
	// This should match the logic used in the main pipeline
	sanitized := strings.ToLower(name)
	sanitized = strings.ReplaceAll(sanitized, " ", "_")
	sanitized = strings.ReplaceAll(sanitized, "/", "_")
	sanitized = strings.ReplaceAll(sanitized, "\\", "_")
	sanitized = strings.ReplaceAll(sanitized, ":", "_")
	sanitized = strings.ReplaceAll(sanitized, "*", "_")
	sanitized = strings.ReplaceAll(sanitized, "?", "_")
	sanitized = strings.ReplaceAll(sanitized, "\"", "_")
	sanitized = strings.ReplaceAll(sanitized, "<", "_")
	sanitized = strings.ReplaceAll(sanitized, ">", "_")
	sanitized = strings.ReplaceAll(sanitized, "|", "_")
	return sanitized
}

func (bp *BatchPipeline) loadProcessedNames() error {
	// Load from data/batch_processed_names.txt
	filePath := filepath.Join("data", "batch_processed_names.txt")

	file, err := os.Open(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // File doesn't exist yet, that's okay
		}
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Parse stored format: "trackingKey|fullName" or just "name" for backwards compatibility
		parts := strings.SplitN(line, "|", 2)
		if len(parts) >= 1 {
			trackingKey := parts[0]
			bp.processedNames[trackingKey] = true

			// Store full name if available
			if len(parts) == 2 {
				bp.processedAliases[trackingKey] = parts[1]
			}
		}
	}

	return scanner.Err()
}

func (bp *BatchPipeline) saveProcessedName(personaName *PersonaName) error {
	// Ensure data directory exists
	if err := os.MkdirAll("data", 0755); err != nil {
		return err
	}

	// Append to data/batch_processed_names.txt
	filePath := filepath.Join("data", "batch_processed_names.txt")
	file, err := os.OpenFile(filePath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	// Save in format: "trackingKey|fullName"
	trackingKey := personaName.GetTrackingKey()
	line := trackingKey
	if personaName.HasAlias() {
		// For aliases, save the full information
		line = fmt.Sprintf("%s|%s", trackingKey, personaName.FullName)
	}

	_, err = fmt.Fprintln(file, line)
	return err
}
