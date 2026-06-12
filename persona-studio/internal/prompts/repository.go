package prompts

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/twin2ai/studio/internal/assets"
)

// RepositoryManager handles file operations in the persona repository
type RepositoryManager struct {
	baseDir       string
	logger        *logrus.Logger
	statusManager *assets.StatusManager
}

// NewRepositoryManager creates a new repository manager
func NewRepositoryManager(baseDir string, logger *logrus.Logger) *RepositoryManager {
	return &RepositoryManager{
		baseDir:       baseDir,
		logger:        logger,
		statusManager: assets.NewStatusManager(baseDir),
	}
}

// SavePromptResults saves all generated prompts to the persona repository
func (rm *RepositoryManager) SavePromptResults(ctx context.Context, personaName string, results []PromptResult) error {
	rm.logger.Infof("Saving %d prompt results for persona: %s", len(results), personaName)

	// Get persona folder path
	personaFolder := rm.getPersonaFolder(personaName)

	// Create prompts subdirectory
	promptsDir := filepath.Join(personaFolder, "prompts")
	if err := os.MkdirAll(promptsDir, 0755); err != nil {
		return fmt.Errorf("failed to create prompts directory: %w", err)
	}

	// Save each prompt result
	savedCount := 0
	for _, result := range results {
		if result.Error != nil {
			rm.logger.Warnf("Skipping failed prompt generation for %s: %v", result.PromptType, result.Error)
			continue
		}

		// Validate the result
		if err := ValidatePromptResult(&result); err != nil {
			rm.logger.Warnf("Skipping invalid prompt result for %s: %v", result.PromptType, err)
			continue
		}

		// Save the prompt file
		if err := rm.savePromptFile(promptsDir, result); err != nil {
			rm.logger.Errorf("Failed to save prompt file for %s: %v", result.PromptType, err)
			continue
		}

		savedCount++
	}

	rm.logger.Infof("Successfully saved %d prompts for %s", savedCount, personaName)

	// Update README to include prompt information
	if err := rm.updatePersonaReadme(personaName, results); err != nil {
		rm.logger.Warnf("Failed to update persona README: %v", err)
	}

	// Update asset status
	if err := rm.updateAssetStatus(personaName, results); err != nil {
		rm.logger.Warnf("Failed to update asset status: %v", err)
	}

	return nil
}

// savePromptFile saves a single prompt result to a file
func (rm *RepositoryManager) savePromptFile(promptsDir string, result PromptResult) error {
	filename := GetPromptFilename(result.PromptType)
	// Remove the "prompts/" prefix since we're already in the prompts directory
	filename = strings.TrimPrefix(filename, "prompts/")

	filePath := filepath.Join(promptsDir, filename)

	// Create the prompt content with metadata
	content := rm.formatPromptContent(result)

	// Write the file
	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		return fmt.Errorf("failed to write prompt file %s: %w", filePath, err)
	}

	rm.logger.Debugf("Saved prompt file: %s", filePath)
	return nil
}

// formatPromptContent formats the prompt content with metadata
func (rm *RepositoryManager) formatPromptContent(result PromptResult) string {
	header := fmt.Sprintf(`# %s

> **Generated:** %s  
> **Persona:** %s  
> **Type:** %s  
> **Source:** synthesized.md  

---

`, GetPromptDisplayName(result.PromptType),
		result.GeneratedAt.Format(time.RFC3339),
		result.PersonaName,
		result.PromptType)

	footer := fmt.Sprintf(`

---

*Generated automatically by [Studio](https://github.com/twin2ai/studio) using Gemini 2.0 Flash*  
*Last updated: %s*
`, result.GeneratedAt.Format("2006-01-02 15:04:05 UTC"))

	return header + result.Content + footer
}

// updatePersonaReadme updates the persona README to include prompt information
func (rm *RepositoryManager) updatePersonaReadme(personaName string, results []PromptResult) error {
	readmePath := filepath.Join(rm.getPersonaFolder(personaName), "README.md")

	// Read existing README
	content, err := os.ReadFile(readmePath)
	if err != nil {
		rm.logger.Warnf("Could not read existing README for update: %v", err)
		return nil // Don't fail the whole operation
	}

	readmeContent := string(content)

	// Check if prompts section already exists
	promptsSectionStart := "## Generated Prompts"
	if strings.Contains(readmeContent, promptsSectionStart) {
		// Replace existing prompts section
		return rm.replacePromptsSection(readmePath, readmeContent, results)
	} else {
		// Add new prompts section
		return rm.addPromptsSection(readmePath, readmeContent, results)
	}
}

// addPromptsSection adds a new prompts section to the README
func (rm *RepositoryManager) addPromptsSection(readmePath, readmeContent string, results []PromptResult) error {
	promptsSection := rm.generatePromptsSection(results)

	// Insert before the "Source" section if it exists, otherwise append
	sourceSection := "## Source"
	if strings.Contains(readmeContent, sourceSection) {
		readmeContent = strings.Replace(readmeContent, sourceSection, promptsSection+"\n"+sourceSection, 1)
	} else {
		// Insert before the final separator if it exists
		finalSeparator := "\n---\n"
		if strings.Contains(readmeContent, finalSeparator) {
			readmeContent = strings.Replace(readmeContent, finalSeparator, "\n"+promptsSection+finalSeparator, 1)
		} else {
			readmeContent += "\n" + promptsSection
		}
	}

	return os.WriteFile(readmePath, []byte(readmeContent), 0644)
}

// replacePromptsSection replaces the existing prompts section in the README
func (rm *RepositoryManager) replacePromptsSection(readmePath, readmeContent string, results []PromptResult) error {
	promptsSection := rm.generatePromptsSection(results)

	// Find the start of the prompts section
	startMarker := "## Generated Prompts"
	startIndex := strings.Index(readmeContent, startMarker)
	if startIndex == -1 {
		return rm.addPromptsSection(readmePath, readmeContent, results)
	}

	// Find the end of the prompts section (next ## heading or end of file)
	afterStart := readmeContent[startIndex+len(startMarker):]
	nextSectionIndex := strings.Index(afterStart, "\n## ")

	var newContent string
	if nextSectionIndex == -1 {
		// No next section, replace to end of file
		newContent = readmeContent[:startIndex] + promptsSection
	} else {
		// Replace until next section
		endIndex := startIndex + len(startMarker) + nextSectionIndex
		newContent = readmeContent[:startIndex] + promptsSection + "\n" + readmeContent[endIndex:]
	}

	return os.WriteFile(readmePath, []byte(newContent), 0644)
}

// generatePromptsSection generates the prompts section for the README
func (rm *RepositoryManager) generatePromptsSection(results []PromptResult) string {
	var platformPrompts []string
	var variationPrompts []string

	for _, result := range results {
		if result.Error != nil {
			continue
		}

		filename := strings.TrimPrefix(GetPromptFilename(result.PromptType), "prompts/")
		displayName := GetPromptDisplayName(result.PromptType)

		promptEntry := fmt.Sprintf("- **[%s](prompts/%s)** - %s", displayName, filename, rm.getPromptDescription(result.PromptType))

		if IsPlatformPrompt(result.PromptType) {
			platformPrompts = append(platformPrompts, promptEntry)
		} else if IsVariationPrompt(result.PromptType) {
			variationPrompts = append(variationPrompts, promptEntry)
		}
	}

	section := "## Generated Prompts\n\n"
	section += fmt.Sprintf("*Last updated: %s*\n\n", time.Now().Format("2006-01-02 15:04:05 UTC"))

	if len(platformPrompts) > 0 {
		section += "### 🎯 Platform-Specific Prompts\n"
		for _, prompt := range platformPrompts {
			section += prompt + "\n"
		}
		section += "\n"
	}

	if len(variationPrompts) > 0 {
		section += "### 📝 Prompt Variations\n"
		for _, prompt := range variationPrompts {
			section += prompt + "\n"
		}
		section += "\n"
	}

	section += "### How to Use\n"
	section += "1. **Copy the prompt** from the appropriate file\n"
	section += "2. **Paste into your AI platform** (ChatGPT, Claude, etc.)\n"
	section += "3. **Start your conversation** - the AI will embody this persona\n"
	section += "4. **Regenerate prompts** by updating the synthesized.md file\n\n"

	return section
}

// getPromptDescription returns a description for each prompt type
func (rm *RepositoryManager) getPromptDescription(promptType PromptType) string {
	switch promptType {
	case PromptTypeChatGPT:
		return "Optimized for ChatGPT's interface and capabilities"
	case PromptTypeClaude:
		return "Tailored for Claude's analytical and helpful nature"
	case PromptTypeGemini:
		return "Designed for Gemini's multimodal and practical approach"
	case PromptTypeDiscord:
		return "Casual bot personality for Discord communities"
	case PromptTypeCharacterAI:
		return "Immersive character for roleplay conversations"
	case PromptTypeCondensed:
		return "Compact version for quick use and copy-paste"
	case PromptTypeAlternative:
		return "Different facets and variations of the persona"
	default:
		return "Custom prompt variation"
	}
}

// updateAssetStatus updates the asset status to reflect prompt generation
func (rm *RepositoryManager) updateAssetStatus(personaName string, results []PromptResult) error {
	// Load current status
	status, err := rm.statusManager.LoadStatus(personaName)
	if err != nil {
		rm.logger.Warnf("Failed to load asset status for prompt update: %v", err)
		return err
	}

	// Update generated assets
	for _, result := range results {
		if result.Error == nil {
			assetName := fmt.Sprintf("prompt_%s", result.PromptType)

			// Remove from pending
			status.PendingAssets = removeFromSlice(status.PendingAssets, assetName)

			// Add to generated
			if !contains(status.GeneratedAssets, assetName) {
				status.GeneratedAssets = append(status.GeneratedAssets, assetName)
			}

			// Set generation flag
			if status.AssetGenerationFlags == nil {
				status.AssetGenerationFlags = make(map[string]bool)
			}
			status.AssetGenerationFlags[assetName] = true
		}
	}

	// Update generation timestamp
	status.LastAssetsGeneration = time.Now()

	// Add metadata
	if status.Metadata == nil {
		status.Metadata = make(map[string]string)
	}
	status.Metadata["last_prompt_generation"] = time.Now().Format(time.RFC3339)
	status.Metadata["prompts_generated"] = fmt.Sprintf("%d", len(results))

	return rm.statusManager.SaveStatus(status)
}

// getPersonaFolder returns the folder path for a persona
func (rm *RepositoryManager) getPersonaFolder(personaName string) string {
	// Normalize persona name to folder format
	folderName := strings.ToLower(strings.ReplaceAll(personaName, " ", "_"))
	folderName = strings.ReplaceAll(folderName, "/", "_")
	return filepath.Join(rm.baseDir, "personas", folderName)
}

// Helper functions
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func removeFromSlice(slice []string, item string) []string {
	var result []string
	for _, s := range slice {
		if s != item {
			result = append(result, s)
		}
	}
	return result
}
