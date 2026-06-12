package synthesizer

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/sirupsen/logrus"
	"github.com/twin2ai/studio/internal/config"
	"github.com/twin2ai/studio/internal/gemini"
	"github.com/twin2ai/studio/internal/github"
)

// Synthesizer handles regenerating synthesized.md from raw AI outputs
type Synthesizer struct {
	config       *config.Config
	githubClient *github.Client
	geminiClient *gemini.Client
	logger       *logrus.Logger
}

// New creates a new synthesizer instance
func New(cfg *config.Config, logger *logrus.Logger) *Synthesizer {
	// Create GitHub client
	githubClient := github.NewClient(
		cfg.GitHub.Token,
		cfg.GitHub.Owner,
		cfg.GitHub.Repo,
		cfg.GitHub.PersonasOwner,
		cfg.GitHub.PersonasRepo,
		cfg.GitHub.PersonaLabel,
		logger,
	)

	// Create Gemini client for synthesis
	geminiClient := gemini.NewClient(cfg.AI.Gemini.APIKey, cfg.AI.Gemini.Model, logger)

	return &Synthesizer{
		config:       cfg,
		githubClient: githubClient,
		geminiClient: geminiClient,
		logger:       logger,
	}
}

// SynthesizeAll regenerates synthesized.md for all personas
func (s *Synthesizer) SynthesizeAll(ctx context.Context) error {
	// List all persona folders from GitHub
	personas, err := s.githubClient.ListPersonaFolders(ctx)
	if err != nil {
		return fmt.Errorf("failed to list personas: %w", err)
	}

	s.logger.Infof("Found %d personas to synthesize", len(personas))

	successCount := 0
	for _, personaFolder := range personas {
		// Convert folder name back to persona name (e.g., "elon_musk" -> "Elon Musk")
		personaName := s.folderToPersonaName(personaFolder)

		s.logger.Infof("Processing persona: %s", personaName)
		if err := s.SynthesizeOne(ctx, personaName); err != nil {
			s.logger.Errorf("Failed to synthesize %s: %v", personaName, err)
			continue
		}
		successCount++
	}

	s.logger.Infof("Successfully synthesized %d/%d personas", successCount, len(personas))
	return nil
}

// SynthesizeOne regenerates synthesized.md for a specific persona
func (s *Synthesizer) SynthesizeOne(ctx context.Context, personaName string) error {
	// Normalize persona name to folder name
	folderName := s.personaToFolderName(personaName)

	// Fetch raw AI outputs from GitHub
	rawOutputs, err := s.fetchRawOutputs(ctx, folderName)
	if err != nil {
		return fmt.Errorf("failed to fetch raw outputs: %w", err)
	}

	// Check if we have sufficient raw outputs
	if len(rawOutputs) == 0 {
		return fmt.Errorf("no raw AI outputs found for persona %s", personaName)
	}

	s.logger.Infof("Found %d raw AI outputs for %s", len(rawOutputs), personaName)

	// Load the combination prompt
	combinationPrompt, err := s.loadCombinationPrompt()
	if err != nil {
		return fmt.Errorf("failed to load combination prompt: %w", err)
	}

	// Prepare the prompt with all raw outputs
	fullPrompt := s.prepareCombinationPrompt(combinationPrompt, rawOutputs, personaName)

	// Generate new synthesis using Gemini with lower temperature for consistency
	s.logger.Info("Generating new synthesis with Gemini...")
	synthesized, err := s.geminiClient.GenerateSynthesis(ctx, fullPrompt)
	if err != nil {
		return fmt.Errorf("failed to generate synthesis: %w", err)
	}

	s.logger.Infof("Generated synthesis with %d characters", len(synthesized))

	// Create a pull request with the updated synthesis
	if err := s.createUpdatePR(ctx, personaName, folderName, synthesized); err != nil {
		return fmt.Errorf("failed to create update PR: %w", err)
	}

	return nil
}

// fetchRawOutputs fetches all raw AI outputs for a persona from GitHub
func (s *Synthesizer) fetchRawOutputs(ctx context.Context, folderName string) (map[string]string, error) {
	outputs := make(map[string]string)

	// List of raw files to fetch
	rawFiles := []struct {
		name string
		path string
	}{
		{"Claude", fmt.Sprintf("personas/%s/raw/claude.md", folderName)},
		{"Gemini", fmt.Sprintf("personas/%s/raw/gemini.md", folderName)},
		{"Grok", fmt.Sprintf("personas/%s/raw/grok.md", folderName)},
		{"GPT-4", fmt.Sprintf("personas/%s/raw/gpt.md", folderName)},
		{"User", fmt.Sprintf("personas/%s/raw/user_supplied.md", folderName)},
	}

	for _, file := range rawFiles {
		content, err := s.githubClient.GetFileContent(ctx, file.path)
		if err != nil {
			// User-supplied is optional, others are required
			if file.name == "User" {
				s.logger.Debugf("No user-supplied persona found (optional)")
				continue
			}
			s.logger.Warnf("Failed to fetch %s output: %v", file.name, err)
			continue
		}
		outputs[file.name] = content
		s.logger.Debugf("Fetched %s output: %d characters", file.name, len(content))
	}

	return outputs, nil
}

// loadCombinationPrompt loads the persona combination prompt template
func (s *Synthesizer) loadCombinationPrompt() (string, error) {
	// Try to load from the prompts directory
	promptPath := filepath.Join("prompts", "persona_combination.txt")
	content, err := os.ReadFile(promptPath)
	if err != nil {
		// Fallback to internal default
		return s.getDefaultCombinationPrompt(), nil
	}
	return string(content), nil
}

// prepareCombinationPrompt prepares the full prompt with all raw outputs
func (s *Synthesizer) prepareCombinationPrompt(template string, rawOutputs map[string]string, personaName string) string {
	// Replace the {{PERSONAS}} placeholder with the actual personas
	personasSection := s.formatPersonasForPrompt(rawOutputs)

	// Replace the placeholder in the template
	fullPrompt := strings.Replace(template, "{{PERSONAS}}", personasSection, 1)

	return fullPrompt
}

// formatPersonasForPrompt formats the raw outputs according to the expected input format
func (s *Synthesizer) formatPersonasForPrompt(rawOutputs map[string]string) string {
	var output strings.Builder

	// Order matters for consistency
	providers := []string{"Claude", "Gemini", "Grok", "GPT-4"}

	for i, provider := range providers {
		if content, exists := rawOutputs[provider]; exists {
			if i > 0 {
				output.WriteString("\n\n")
			}
			output.WriteString(fmt.Sprintf("Persona %d: %s\n", i+1, provider))
			output.WriteString("<<<\n")
			output.WriteString(content)
			output.WriteString("\n>>>")
		}
	}

	// Add user-supplied persona if it exists
	if userContent, exists := rawOutputs["User"]; exists {
		output.WriteString("\n\nUser-Supplied Persona\n")
		output.WriteString("<<<\n")
		output.WriteString(userContent)
		output.WriteString("\n>>>")
	}

	return output.String()
}

// createUpdatePR creates a pull request with the updated synthesized.md
func (s *Synthesizer) createUpdatePR(ctx context.Context, personaName, folderName string, synthesized string) error {
	pr, err := s.githubClient.CreateSynthesisUpdatePR(ctx, personaName, folderName, synthesized)
	if err != nil {
		return fmt.Errorf("failed to create synthesis update PR: %w", err)
	}

	s.logger.Infof("Successfully created PR #%d: %s", pr.GetNumber(), pr.GetHTMLURL())
	return nil
}

// personaToFolderName converts persona name to folder name (e.g., "Elon Musk" -> "elon_musk")
func (s *Synthesizer) personaToFolderName(personaName string) string {
	folderName := strings.ToLower(strings.ReplaceAll(personaName, " ", "_"))
	folderName = strings.ReplaceAll(folderName, "/", "_")
	return folderName
}

// folderToPersonaName converts folder name to persona name (e.g., "elon_musk" -> "Elon Musk")
func (s *Synthesizer) folderToPersonaName(folderName string) string {
	// Simple title case conversion - this won't be perfect for all names
	// but it's a reasonable approximation
	words := strings.Split(folderName, "_")
	for i, word := range words {
		if len(word) > 0 {
			words[i] = strings.ToUpper(word[:1]) + word[1:]
		}
	}
	return strings.Join(words, " ")
}

// getDefaultCombinationPrompt returns the default combination prompt
func (s *Synthesizer) getDefaultCombinationPrompt() string {
	return `You are tasked with synthesizing multiple AI-generated personas into a single, comprehensive persona. Each AI has created their own interpretation based on the same initial requirements.

Your goal is to:

1. **Identify Common Themes**: Find the core characteristics, behaviors, and traits that appear across multiple AI interpretations
2. **Capture Unique Insights**: Preserve valuable unique elements that only one AI might have identified
3. **Resolve Contradictions**: Where AIs disagree, make thoughtful decisions about which interpretation best serves the persona
4. **Enhance Depth**: Combine complementary details to create a richer, more nuanced character
5. **Ensure Coherence**: Create a unified persona that feels like a single, well-developed character

Guidelines:
- Maintain the essential identity and purpose of the persona
- Prioritize elements that appear in multiple versions
- Include specific examples, quotes, and behavioral patterns
- Ensure the voice and communication style is consistent
- Create a persona that is immediately usable in applications

The synthesized persona should be comprehensive (3000-5000 words) and include all necessary details for someone to effectively roleplay or implement this character.`
}
