package prompts

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/twin2ai/studio/internal/assets"
	"github.com/twin2ai/studio/internal/gemini"
	"github.com/twin2ai/studio/internal/github"
)

// GitHubService handles prompt generation with GitHub PR integration
type GitHubService struct {
	promptService *Service
	githubClient  *github.Client
	logger        *logrus.Logger
}

// NewGitHubService creates a new GitHub-integrated prompt service
func NewGitHubService(geminiClient *gemini.Client, githubClient *github.Client, logger *logrus.Logger, baseDir string) *GitHubService {
	promptService := NewService(geminiClient, logger, baseDir)

	return &GitHubService{
		promptService: promptService,
		githubClient:  githubClient,
		logger:        logger,
	}
}

// GenerateAllPromptsWithPR generates all prompts and creates a GitHub PR
func (gs *GitHubService) GenerateAllPromptsWithPR(ctx context.Context, personaName string, assetType assets.AssetType) error {
	return gs.generatePromptsWithPR(ctx, personaName, assetType, "all", false)
}

// GenerateAllPromptsWithPRForced generates all prompts and creates a GitHub PR, bypassing existing checks
func (gs *GitHubService) GenerateAllPromptsWithPRForced(ctx context.Context, personaName string, assetType assets.AssetType) error {
	return gs.generatePromptsWithPR(ctx, personaName, assetType, "all", true)
}

// GeneratePlatformPromptsWithPR generates platform prompts and creates a GitHub PR
func (gs *GitHubService) GeneratePlatformPromptsWithPR(ctx context.Context, personaName string, assetType assets.AssetType) error {
	return gs.generatePromptsWithPR(ctx, personaName, assetType, "platform", false)
}

// GenerateVariationPromptsWithPR generates variation prompts and creates a GitHub PR
func (gs *GitHubService) GenerateVariationPromptsWithPR(ctx context.Context, personaName string, assetType assets.AssetType) error {
	return gs.generatePromptsWithPR(ctx, personaName, assetType, "variation", false)
}

// GeneratePlatformPromptsWithPRForced generates platform prompts and creates a GitHub PR, bypassing existing checks
func (gs *GitHubService) GeneratePlatformPromptsWithPRForced(ctx context.Context, personaName string, assetType assets.AssetType) error {
	return gs.generatePromptsWithPR(ctx, personaName, assetType, "platform", true)
}

// GenerateVariationPromptsWithPRForced generates variation prompts and creates a GitHub PR, bypassing existing checks
func (gs *GitHubService) GenerateVariationPromptsWithPRForced(ctx context.Context, personaName string, assetType assets.AssetType) error {
	return gs.generatePromptsWithPR(ctx, personaName, assetType, "variation", true)
}

// generatePromptsWithPR is the core method that generates prompts and creates a PR
func (gs *GitHubService) generatePromptsWithPR(ctx context.Context, personaName string, assetType assets.AssetType, promptType string, force bool) error {
	gs.logger.Infof("Generating %s prompts with PR for persona: %s (force: %v)", promptType, personaName, force)

	// Read the synthesized persona content from GitHub
	synthesizedContent, err := gs.fetchSynthesizedFromGitHub(ctx, personaName)
	if err != nil {
		return fmt.Errorf("failed to fetch synthesized persona from GitHub: %w", err)
	}

	// Check if prompts already exist in the repo (unless forced)
	if !force && gs.promptsAlreadyExist(ctx, personaName) {
		gs.logger.Infof("Prompts already exist for %s in GitHub repo, skipping generation", personaName)
		return nil
	}

	// Check if we should create a new PR using PR tracker
	prTracker := NewPRTracker(gs.promptService.baseDir, gs.logger)
	shouldCreate, reason, err := prTracker.ShouldCreatePR(ctx, personaName, string(synthesizedContent), gs.githubClient)
	if err != nil {
		gs.logger.Warnf("Failed to check PR tracking status: %v", err)
		// Continue with PR creation despite tracking error
	} else if !shouldCreate {
		gs.logger.Infof("Skipping PR creation for %s: %s", personaName, reason)
		return nil
	}

	// Generate prompts based on type
	var results []PromptResult
	switch promptType {
	case "all":
		results, err = gs.promptService.generator.GenerateAllPrompts(ctx, personaName, string(synthesizedContent))
	case "platform":
		results, err = gs.generatePlatformPrompts(ctx, personaName, string(synthesizedContent))
	case "variation":
		results, err = gs.generateVariationPrompts(ctx, personaName, string(synthesizedContent))
	default:
		return fmt.Errorf("unknown prompt type: %s", promptType)
	}

	if err != nil {
		return fmt.Errorf("failed to generate prompts: %w", err)
	}

	// Skip local saving for GitHub service - we only create PRs
	// The prompts will be saved in the GitHub PR instead of locally
	gs.logger.Debugf("Skipping local file saving for GitHub PR workflow")

	// Prepare data for GitHub PR
	prData, err := gs.preparePRData(personaName, results)
	if err != nil {
		return fmt.Errorf("failed to prepare PR data: %w", err)
	}

	// Create GitHub pull request
	pr, err := gs.githubClient.CreatePromptUpdatePR(ctx, *prData)
	if err != nil {
		return fmt.Errorf("failed to create GitHub PR: %w", err)
	}

	// Track the created PR to prevent duplicates
	contentHash := prTracker.GetContentHash(string(synthesizedContent))
	if err := prTracker.TrackPR(personaName, pr.GetNumber(), pr.GetHTMLURL(), contentHash); err != nil {
		gs.logger.Warnf("Failed to track PR for %s: %v", personaName, err)
		// Don't fail the entire operation for tracking errors
	}

	gs.logger.Infof("Successfully created prompt update PR for %s: %s", personaName, pr.GetHTMLURL())
	return nil
}

// generatePlatformPrompts generates only platform-specific prompts
func (gs *GitHubService) generatePlatformPrompts(ctx context.Context, personaName, synthesizedContent string) ([]PromptResult, error) {
	platformTypes := []PromptType{
		PromptTypeChatGPT,
		PromptTypeClaude,
		PromptTypeGemini,
		PromptTypeDiscord,
		PromptTypeCharacterAI,
	}

	var results []PromptResult
	for _, promptType := range platformTypes {
		result, err := gs.promptService.generator.GeneratePrompt(ctx, personaName, synthesizedContent, promptType)
		if err != nil {
			gs.logger.Errorf("Failed to generate %s prompt: %v", promptType, err)
			// Add error result to track failures
			results = append(results, PromptResult{
				PromptType:  promptType,
				PersonaName: personaName,
				Error:       err,
			})
			continue
		}
		results = append(results, *result)
	}

	return results, nil
}

// generateVariationPrompts generates only variation prompts
func (gs *GitHubService) generateVariationPrompts(ctx context.Context, personaName, synthesizedContent string) ([]PromptResult, error) {
	variationTypes := []PromptType{
		PromptTypeCondensed,
		PromptTypeAlternative,
	}

	var results []PromptResult
	for _, promptType := range variationTypes {
		result, err := gs.promptService.generator.GeneratePrompt(ctx, personaName, synthesizedContent, promptType)
		if err != nil {
			gs.logger.Errorf("Failed to generate %s prompt: %v", promptType, err)
			// Add error result to track failures
			results = append(results, PromptResult{
				PromptType:  promptType,
				PersonaName: personaName,
				Error:       err,
			})
			continue
		}
		results = append(results, *result)
	}

	return results, nil
}

// preparePRData prepares the data structure needed for GitHub PR creation
func (gs *GitHubService) preparePRData(personaName string, results []PromptResult) (*github.PromptPRData, error) {
	// Convert prompt results to GitHub format
	githubResults := gs.convertPromptResults(results)
	// Load updated README content (if it was updated)
	readmeContent, err := gs.loadUpdatedReadme(personaName)
	if err != nil {
		gs.logger.Warnf("Failed to load updated README: %v", err)
		readmeContent = "" // Continue without README update
	}

	// Load updated asset status
	statusContent, err := gs.loadUpdatedAssetStatus(personaName, results)
	if err != nil {
		gs.logger.Warnf("Failed to load updated asset status: %v", err)
		statusContent = "" // Continue without status update
	}

	return &github.PromptPRData{
		PersonaName:   personaName,
		PromptResults: githubResults,
		UpdatedReadme: readmeContent,
		UpdatedStatus: statusContent,
		IssueNumber:   nil, // TODO: Could be passed in if this is related to an issue
		IsUpdate:      gs.checkIfPersonaExists(personaName),
	}, nil
}

// loadUpdatedReadme generates or updates the README content to include prompt files
func (gs *GitHubService) loadUpdatedReadme(personaName string) (string, error) {
	// Try to fetch existing README from GitHub first
	folderName := gs.normalizePersonaName(personaName)
	readmePath := fmt.Sprintf("personas/%s/README.md", folderName)

	ctx := context.Background()
	existingContent, err := gs.githubClient.GetFileContent(ctx, readmePath)
	if err != nil {
		gs.logger.Debugf("README not found in GitHub, will create new one: %v", err)
		// Generate new README
		return gs.generateReadmeContent(personaName, ""), nil
	}

	// Update existing README to include prompts section
	updatedContent := gs.updateReadmeWithPrompts(existingContent, personaName)
	return updatedContent, nil
}

// loadUpdatedAssetStatus generates or updates the asset status to mark prompt assets as generated
func (gs *GitHubService) loadUpdatedAssetStatus(personaName string, promptResults []PromptResult) (string, error) {
	// Try to fetch existing asset status from GitHub first
	folderName := gs.normalizePersonaName(personaName)
	statusPath := fmt.Sprintf("personas/%s/.assets_status.json", folderName)

	ctx := context.Background()
	existingContent, err := gs.githubClient.GetFileContent(ctx, statusPath)
	if err != nil {
		gs.logger.Debugf("Asset status not found in GitHub, will create new one: %v", err)
		// Generate new asset status
		return gs.generateAssetStatus(personaName, promptResults), nil
	}

	// Update existing asset status to include prompt generation
	updatedContent, err := gs.updateAssetStatus(existingContent, personaName, promptResults)
	if err != nil {
		gs.logger.Warnf("Failed to update existing asset status, creating new one: %v", err)
		return gs.generateAssetStatus(personaName, promptResults), nil
	}

	return updatedContent, nil
}

// checkIfPersonaExists checks if the persona already exists in the repository
func (gs *GitHubService) checkIfPersonaExists(personaName string) bool {
	// This is a simple check - in a real implementation, you might want to
	// check the actual GitHub repository
	personaPath := gs.promptService.getPersonaFolder(personaName)
	if _, err := os.Stat(personaPath); os.IsNotExist(err) {
		return false
	}
	return true
}

// RegisterCallbacks registers GitHub-integrated prompt generation callbacks
func (gs *GitHubService) RegisterCallbacks(monitor *assets.Monitor) {
	// Register callbacks that create GitHub PRs
	monitor.RegisterCallback(assets.AssetTypePrompts, gs.GenerateAllPromptsWithPR)
	monitor.RegisterCallback(assets.AssetTypePlatformPrompts, gs.GeneratePlatformPromptsWithPR)
	monitor.RegisterCallback(assets.AssetTypeVariationPrompts, gs.GenerateVariationPromptsWithPR)

	gs.logger.Info("Registered GitHub-integrated prompt generation callbacks")
}

// TriggerPromptGenerationWithPR manually triggers prompt generation with PR creation
func (gs *GitHubService) TriggerPromptGenerationWithPR(ctx context.Context, personaName string, issueNumber *int) error {
	return gs.triggerPromptGenerationWithPR(ctx, personaName, issueNumber, false)
}

// TriggerPromptGenerationWithPRForced manually triggers prompt generation with PR creation, bypassing existing checks
func (gs *GitHubService) TriggerPromptGenerationWithPRForced(ctx context.Context, personaName string, issueNumber *int) error {
	return gs.triggerPromptGenerationWithPR(ctx, personaName, issueNumber, true)
}

// triggerPromptGenerationWithPR is the internal method for triggering prompt generation
func (gs *GitHubService) triggerPromptGenerationWithPR(ctx context.Context, personaName string, issueNumber *int, force bool) error {
	gs.logger.Infof("Manually triggering prompt generation with PR for: %s (force: %v)", personaName, force)

	// Mark prompts as pending in asset status
	statusManager := assets.NewStatusManager(gs.promptService.baseDir)

	promptAssets := []assets.AssetType{
		assets.AssetTypePrompts,
		assets.AssetTypePlatformPrompts,
		assets.AssetTypeVariationPrompts,
	}

	for _, assetType := range promptAssets {
		if err := statusManager.MarkAssetPending(personaName, assetType); err != nil {
			gs.logger.Warnf("Failed to mark %s as pending: %v", assetType, err)
		}
	}

	// Generate prompts with PR creation
	if force {
		return gs.GenerateAllPromptsWithPRForced(ctx, personaName, assets.AssetTypePrompts)
	}
	return gs.GenerateAllPromptsWithPR(ctx, personaName, assets.AssetTypePrompts)
}

// GetPromptGenerationStats returns statistics (delegated to the base service)
func (gs *GitHubService) GetPromptGenerationStats(personaName string) (map[string]interface{}, error) {
	return gs.promptService.GetPromptGenerationStats(personaName)
}

// convertPromptResults converts prompts.PromptResult to github.PromptResult
func (gs *GitHubService) convertPromptResults(results []PromptResult) []github.PromptResult {
	var githubResults []github.PromptResult

	for _, result := range results {
		githubResult := github.PromptResult{
			PromptType:  string(result.PromptType),
			Content:     result.Content,
			GeneratedAt: result.GeneratedAt,
			PersonaName: result.PersonaName,
			Error:       result.Error,
		}
		githubResults = append(githubResults, githubResult)
	}

	return githubResults
}

// fetchSynthesizedFromGitHub fetches the synthesized.md content from the GitHub repository
func (gs *GitHubService) fetchSynthesizedFromGitHub(ctx context.Context, personaName string) (string, error) {
	// Normalize persona name for folder structure
	folderName := gs.normalizePersonaName(personaName)

	// Construct path to synthesized.md in the personas repo
	filePath := fmt.Sprintf("personas/%s/synthesized.md", folderName)

	gs.logger.Debugf("Fetching synthesized content from GitHub: %s", filePath)

	// Use GitHub client to fetch the file content
	content, err := gs.githubClient.GetFileContent(ctx, filePath)
	if err != nil {
		return "", fmt.Errorf("failed to fetch %s from GitHub: %w", filePath, err)
	}

	return content, nil
}

// normalizePersonaName converts persona name to folder-safe format
func (gs *GitHubService) normalizePersonaName(name string) string {
	// Convert to lowercase and replace spaces with underscores
	// This should match the logic used in structured_pr.go
	folderName := strings.ToLower(strings.ReplaceAll(name, " ", "_"))
	folderName = strings.ReplaceAll(folderName, "/", "_")
	return folderName
}

// generateReadmeContent creates a new README.md content with prompts section
func (gs *GitHubService) generateReadmeContent(personaName string, existingContent string) string {
	var readme strings.Builder

	// If there's existing content, preserve it first
	if existingContent != "" {
		readme.WriteString(existingContent)
		readme.WriteString("\n\n")
	} else {
		// Create basic README structure for new personas
		readme.WriteString(fmt.Sprintf("# %s\n\n", personaName))
		readme.WriteString("This persona was generated by the [Studio Multi-AI Persona Generation Pipeline](https://github.com/twin2ai/studio).\n\n")

		// Check if synthesized.md exists to add description
		ctx := context.Background()
		folderName := gs.normalizePersonaName(personaName)
		synthesizedPath := fmt.Sprintf("personas/%s/synthesized.md", folderName)
		synthesizedContent, err := gs.githubClient.GetFileContent(ctx, synthesizedPath)
		if err == nil && len(synthesizedContent) > 200 {
			// Add first paragraph as description
			lines := strings.Split(synthesizedContent, "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if len(line) > 50 && !strings.HasPrefix(line, "#") {
					readme.WriteString("## Description\n\n")
					readme.WriteString(line)
					readme.WriteString("\n\n")
					break
				}
			}
		}
	}

	// Add or update the prompts section
	readme.WriteString(gs.generatePromptsSection())

	return readme.String()
}

// updateReadmeWithPrompts updates existing README content to include/update prompts section
func (gs *GitHubService) updateReadmeWithPrompts(existingContent string, personaName string) string {
	// Check if prompts section already exists
	if strings.Contains(existingContent, "## Generated Prompts") || strings.Contains(existingContent, "## Prompts") {
		// Replace existing prompts section
		lines := strings.Split(existingContent, "\n")
		var updatedLines []string
		skipSection := false

		for _, line := range lines {
			// Check if this line starts a prompts section
			if strings.HasPrefix(line, "## Generated Prompts") || strings.HasPrefix(line, "## Prompts") {
				skipSection = true
				// Add updated prompts section
				promptsSection := strings.Split(gs.generatePromptsSection(), "\n")
				updatedLines = append(updatedLines, promptsSection...)
				continue
			}

			// Check if we've reached the next section (and should stop skipping)
			if skipSection && strings.HasPrefix(line, "## ") &&
				!strings.HasPrefix(line, "## Generated Prompts") &&
				!strings.HasPrefix(line, "## Prompts") {
				skipSection = false
			}

			// Only add lines if we're not in the prompts section
			if !skipSection {
				updatedLines = append(updatedLines, line)
			}
		}

		return strings.Join(updatedLines, "\n")
	} else {
		// Add prompts section to existing content
		return gs.generateReadmeContent(personaName, existingContent)
	}
}

// generatePromptsSection creates the prompts section for README
func (gs *GitHubService) generatePromptsSection() string {
	var section strings.Builder

	section.WriteString("## Generated Prompts\n\n")
	section.WriteString("This persona includes optimized prompts for various AI platforms and use cases:\n\n")

	section.WriteString("### Platform-Specific Prompts\n\n")
	section.WriteString("| Platform | File | Description |\n")
	section.WriteString("|----------|------|-------------|\n")
	section.WriteString("| **ChatGPT** | [`prompts/chatgpt.md`](prompts/chatgpt.md) | Optimized for ChatGPT's conversational interface |\n")
	section.WriteString("| **Claude** | [`prompts/claude.md`](prompts/claude.md) | Tailored for Claude's analytical and helpful nature |\n")
	section.WriteString("| **Gemini** | [`prompts/gemini.md`](prompts/gemini.md) | Designed for Google's Gemini AI assistant |\n")
	section.WriteString("| **Discord Bot** | [`prompts/discord.md`](prompts/discord.md) | Formatted for Discord bot personality integration |\n")
	section.WriteString("| **Character.AI** | [`prompts/characterai.md`](prompts/characterai.md) | Structured for Character.AI's roleplay environment |\n\n")

	section.WriteString("### Prompt Variations\n\n")
	section.WriteString("| Type | File | Description |\n")
	section.WriteString("|------|------|-------------|\n")
	section.WriteString("| **Condensed** | [`prompts/condensed.md`](prompts/condensed.md) | Shorter version for quick copy-paste use |\n")
	section.WriteString("| **Alternative** | [`prompts/alternative.md`](prompts/alternative.md) | Alternative approaches and variations |\n\n")

	section.WriteString("### How to Use\n\n")
	section.WriteString("1. **Choose the appropriate prompt** for your AI platform\n")
	section.WriteString("2. **Copy the entire content** from the corresponding file\n")
	section.WriteString("3. **Paste into your AI interface** as a system prompt or initial message\n")
	section.WriteString("4. **Start conversing** with the persona\n\n")

	section.WriteString("### Generation Details\n\n")
	section.WriteString("- **Generated by:** [Studio Multi-AI Pipeline](https://github.com/twin2ai/studio)\n")
	section.WriteString("- **Model:** Gemini 2.0 Flash\n")
	section.WriteString("- **Source:** synthesized.md (combined from 4 AI providers)\n")
	section.WriteString("- **Optimization:** Platform-specific templates and requirements\n\n")

	return section.String()
}

// promptsAlreadyExist checks if prompt files already exist in the GitHub repository
func (gs *GitHubService) promptsAlreadyExist(ctx context.Context, personaName string) bool {
	folderName := gs.normalizePersonaName(personaName)

	// Check for a few key prompt files to determine if prompts already exist
	promptFiles := []string{
		fmt.Sprintf("personas/%s/prompts/chatgpt.md", folderName),
		fmt.Sprintf("personas/%s/prompts/claude.md", folderName),
		fmt.Sprintf("personas/%s/prompts/gemini.md", folderName),
	}

	existingCount := 0
	for _, filePath := range promptFiles {
		_, err := gs.githubClient.GetFileContent(ctx, filePath)
		if err == nil {
			existingCount++
		}
	}

	// If at least 2 out of 3 key prompts exist, consider prompts already generated
	if existingCount >= 2 {
		gs.logger.Debugf("Found %d existing prompt files for %s", existingCount, personaName)
		return true
	}

	gs.logger.Debugf("Found only %d existing prompt files for %s, will generate prompts", existingCount, personaName)
	return false
}

// generateAssetStatus creates a new asset status JSON for a persona with prompt assets marked as generated
func (gs *GitHubService) generateAssetStatus(personaName string, promptResults []PromptResult) string {
	now := time.Now()

	// Determine which prompt assets were successfully generated
	var generatedAssets []string
	var pendingAssets []string
	assetFlags := make(map[string]bool)

	// Check which prompt types were successfully generated
	for _, result := range promptResults {
		if result.Error == nil {
			// Map prompt types to asset types
			switch result.PromptType {
			case PromptTypeChatGPT, PromptTypeClaude, PromptTypeGemini, PromptTypeDiscord, PromptTypeCharacterAI:
				if !contains(generatedAssets, string(assets.AssetTypePlatformPrompts)) {
					generatedAssets = append(generatedAssets, string(assets.AssetTypePlatformPrompts))
					assetFlags[string(assets.AssetTypePlatformPrompts)] = true
				}
			case PromptTypeCondensed, PromptTypeAlternative:
				if !contains(generatedAssets, string(assets.AssetTypeVariationPrompts)) {
					generatedAssets = append(generatedAssets, string(assets.AssetTypeVariationPrompts))
					assetFlags[string(assets.AssetTypeVariationPrompts)] = true
				}
			}
		}
	}

	// If we have any generated prompts, mark the general prompts asset as generated too
	if len(generatedAssets) > 0 {
		generatedAssets = append(generatedAssets, string(assets.AssetTypePrompts))
		assetFlags[string(assets.AssetTypePrompts)] = true
	}

	// Create asset status
	status := &assets.AssetStatus{
		PersonaName:           personaName,
		LastSynthesizedUpdate: now,
		LastAssetsGeneration:  now,
		PendingAssets:         pendingAssets,
		GeneratedAssets:       generatedAssets,
		AssetGenerationFlags:  assetFlags,
		Metadata: map[string]string{
			"generator": "studio-prompt-pipeline",
			"model":     "gemini-2.5-flash",
			"version":   "1.0",
		},
	}

	// Serialize to JSON
	data, err := json.MarshalIndent(status, "", "  ")
	if err != nil {
		gs.logger.Errorf("Failed to marshal asset status: %v", err)
		return "{}" // Return empty JSON on error
	}

	return string(data)
}

// updateAssetStatus updates existing asset status JSON to include new prompt generation
func (gs *GitHubService) updateAssetStatus(existingContent string, personaName string, promptResults []PromptResult) (string, error) {
	// Parse existing asset status
	var status assets.AssetStatus
	if err := json.Unmarshal([]byte(existingContent), &status); err != nil {
		return "", fmt.Errorf("failed to parse existing asset status: %w", err)
	}

	now := time.Now()

	// Initialize maps if nil
	if status.AssetGenerationFlags == nil {
		status.AssetGenerationFlags = make(map[string]bool)
	}

	// Determine which prompt assets were successfully generated
	var newGeneratedAssets []string

	// Check which prompt types were successfully generated
	platformGenerated := false
	variationGenerated := false

	for _, result := range promptResults {
		if result.Error == nil {
			switch result.PromptType {
			case PromptTypeChatGPT, PromptTypeClaude, PromptTypeGemini, PromptTypeDiscord, PromptTypeCharacterAI:
				platformGenerated = true
			case PromptTypeCondensed, PromptTypeAlternative:
				variationGenerated = true
			}
		}
	}

	// Update asset lists and flags
	if platformGenerated {
		assetStr := string(assets.AssetTypePlatformPrompts)
		if !contains(status.GeneratedAssets, assetStr) {
			status.GeneratedAssets = append(status.GeneratedAssets, assetStr)
		}
		status.PendingAssets = removeFromSlice(status.PendingAssets, assetStr)
		status.AssetGenerationFlags[assetStr] = true
		newGeneratedAssets = append(newGeneratedAssets, assetStr)
	}

	if variationGenerated {
		assetStr := string(assets.AssetTypeVariationPrompts)
		if !contains(status.GeneratedAssets, assetStr) {
			status.GeneratedAssets = append(status.GeneratedAssets, assetStr)
		}
		status.PendingAssets = removeFromSlice(status.PendingAssets, assetStr)
		status.AssetGenerationFlags[assetStr] = true
		newGeneratedAssets = append(newGeneratedAssets, assetStr)
	}

	// If we generated any prompts, mark the general prompts asset as generated too
	if len(newGeneratedAssets) > 0 {
		assetStr := string(assets.AssetTypePrompts)
		if !contains(status.GeneratedAssets, assetStr) {
			status.GeneratedAssets = append(status.GeneratedAssets, assetStr)
		}
		status.PendingAssets = removeFromSlice(status.PendingAssets, assetStr)
		status.AssetGenerationFlags[assetStr] = true

		// Update timestamps
		status.LastAssetsGeneration = now
	}

	// Update metadata
	if status.Metadata == nil {
		status.Metadata = make(map[string]string)
	}
	status.Metadata["last_prompt_generation"] = now.Format(time.RFC3339)
	status.Metadata["prompt_generator"] = "studio-prompt-pipeline"
	status.Metadata["prompt_model"] = "gemini-2.5-flash"

	// Serialize to JSON
	data, err := json.MarshalIndent(status, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal updated asset status: %w", err)
	}

	return string(data), nil
}
