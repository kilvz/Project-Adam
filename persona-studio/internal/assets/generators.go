package assets

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/sirupsen/logrus"
)

// GeneratorService provides asset generation capabilities
type GeneratorService struct {
	baseDir string
	logger  *logrus.Logger
}

// NewGeneratorService creates a new generator service
func NewGeneratorService(baseDir string, logger *logrus.Logger) *GeneratorService {
	return &GeneratorService{
		baseDir: baseDir,
		logger:  logger,
	}
}

// GeneratePromptReady generates a prompt-ready version of the persona
func (g *GeneratorService) GeneratePromptReady(ctx context.Context, personaName string, assetType AssetType) error {
	g.logger.Infof("Generating prompt-ready asset for persona: %s", personaName)

	// Read the synthesized persona content
	synthesizedPath := filepath.Join(g.baseDir, "personas", personaName, "synthesized.md")
	content, err := os.ReadFile(synthesizedPath)
	if err != nil {
		return fmt.Errorf("failed to read synthesized persona: %w", err)
	}

	// TODO: Implement actual prompt-ready generation logic
	// For now, create a placeholder
	promptReadyContent := fmt.Sprintf(`# %s - Prompt Ready Version

> **Generated on:** %s  
> **Source:** synthesized.md  
> **Purpose:** Condensed version optimized for AI prompts  

## Quick Overview
This is a placeholder for the prompt-ready version of %s.

## Original Content Preview
%s

---
*This is a placeholder asset. Implement actual prompt-ready generation logic in generators.go*

<!-- ASSET_STATUS:generated -->
<!-- GENERATED_AT:%s -->
`, personaName, time.Now().Format(time.RFC3339), personaName,
		truncateContent(string(content), 500), time.Now().Format(time.RFC3339))

	// Write the prompt-ready file
	outputPath := filepath.Join(g.baseDir, "personas", personaName, "prompt_ready.md")
	if err := os.WriteFile(outputPath, []byte(promptReadyContent), 0644); err != nil {
		return fmt.Errorf("failed to write prompt-ready file: %w", err)
	}

	g.logger.Infof("Successfully generated prompt-ready asset: %s", outputPath)
	return nil
}

// GeneratePlatformAdaptations generates platform-specific adaptations
func (g *GeneratorService) GeneratePlatformAdaptations(ctx context.Context, personaName string, assetType AssetType) error {
	g.logger.Infof("Generating platform adaptations for persona: %s", personaName)

	// Create platforms directory
	platformsDir := filepath.Join(g.baseDir, "personas", personaName, "platforms")
	if err := os.MkdirAll(platformsDir, 0755); err != nil {
		return fmt.Errorf("failed to create platforms directory: %w", err)
	}

	// Generate placeholder adaptations for different platforms
	platforms := []string{"discord", "telegram", "slack", "chatgpt", "character_ai"}

	for _, platform := range platforms {
		content := fmt.Sprintf(`# %s - %s Platform Adaptation

> **Generated on:** %s  
> **Platform:** %s  
> **Source:** synthesized.md  

## Platform-Specific Configuration
This is a placeholder for %s platform adaptation.

### Key Adaptations
- Platform-specific formatting
- Character limits consideration
- Feature availability adjustments
- User experience optimization

---
*This is a placeholder asset. Implement actual platform adaptation logic in generators.go*

<!-- ASSET_STATUS:generated -->
<!-- GENERATED_AT:%s -->
`, personaName, platform, time.Now().Format(time.RFC3339), platform,
			platform, time.Now().Format(time.RFC3339))

		outputPath := filepath.Join(platformsDir, fmt.Sprintf("%s.md", platform))
		if err := os.WriteFile(outputPath, []byte(content), 0644); err != nil {
			g.logger.Warnf("Failed to write platform adaptation for %s: %v", platform, err)
			continue
		}
	}

	g.logger.Infof("Successfully generated platform adaptations in: %s", platformsDir)
	return nil
}

// GenerateVoiceClone generates voice cloning configuration
func (g *GeneratorService) GenerateVoiceClone(ctx context.Context, personaName string, assetType AssetType) error {
	g.logger.Infof("Generating voice clone configuration for persona: %s", personaName)

	content := fmt.Sprintf(`# %s - Voice Clone Configuration

> **Generated on:** %s  
> **Purpose:** Voice synthesis and cloning parameters  
> **Source:** synthesized.md  

## Voice Characteristics
This is a placeholder for voice cloning configuration.

### Recommended Settings
- **Tone:** [Placeholder]
- **Pace:** [Placeholder]  
- **Accent:** [Placeholder]
- **Emotion:** [Placeholder]

### Technical Parameters
- **Pitch Range:** [Placeholder]
- **Speaking Rate:** [Placeholder]
- **Voice Model:** [Placeholder]

---
*This is a placeholder asset. Implement actual voice clone generation logic in generators.go*

<!-- ASSET_STATUS:generated -->
<!-- GENERATED_AT:%s -->
`, personaName, time.Now().Format(time.RFC3339), time.Now().Format(time.RFC3339))

	outputPath := filepath.Join(g.baseDir, "personas", personaName, "voice_clone.md")
	if err := os.WriteFile(outputPath, []byte(content), 0644); err != nil {
		return fmt.Errorf("failed to write voice clone file: %w", err)
	}

	g.logger.Infof("Successfully generated voice clone configuration: %s", outputPath)
	return nil
}

// GenerateImageAvatar generates image avatar configuration
func (g *GeneratorService) GenerateImageAvatar(ctx context.Context, personaName string, assetType AssetType) error {
	g.logger.Infof("Generating image avatar configuration for persona: %s", personaName)

	content := fmt.Sprintf(`# %s - Image Avatar Configuration

> **Generated on:** %s  
> **Purpose:** Avatar and image generation parameters  
> **Source:** synthesized.md  

## Visual Characteristics
This is a placeholder for image avatar configuration.

### Appearance Description
- **Style:** [Placeholder]
- **Age:** [Placeholder]
- **Features:** [Placeholder]
- **Clothing:** [Placeholder]

### Generation Parameters
- **Art Style:** [Placeholder]
- **Resolution:** [Placeholder]
- **Color Palette:** [Placeholder]

---
*This is a placeholder asset. Implement actual image avatar generation logic in generators.go*

<!-- ASSET_STATUS:generated -->
<!-- GENERATED_AT:%s -->
`, personaName, time.Now().Format(time.RFC3339), time.Now().Format(time.RFC3339))

	outputPath := filepath.Join(g.baseDir, "personas", personaName, "image_avatar.md")
	if err := os.WriteFile(outputPath, []byte(content), 0644); err != nil {
		return fmt.Errorf("failed to write image avatar file: %w", err)
	}

	g.logger.Infof("Successfully generated image avatar configuration: %s", outputPath)
	return nil
}

// GenerateChatbotConfig generates chatbot configuration
func (g *GeneratorService) GenerateChatbotConfig(ctx context.Context, personaName string, assetType AssetType) error {
	g.logger.Infof("Generating chatbot configuration for persona: %s", personaName)

	content := fmt.Sprintf(`# %s - Chatbot Configuration

> **Generated on:** %s  
> **Purpose:** Chatbot deployment and configuration  
> **Source:** synthesized.md  

## Chatbot Settings
This is a placeholder for chatbot configuration.

### Behavior Parameters
- **Response Style:** [Placeholder]
- **Context Window:** [Placeholder]
- **Temperature:** [Placeholder]
- **Max Tokens:** [Placeholder]

### Integration Points
- **Platform APIs:** [Placeholder]
- **Webhook URLs:** [Placeholder]
- **Authentication:** [Placeholder]

---
*This is a placeholder asset. Implement actual chatbot config generation logic in generators.go*

<!-- ASSET_STATUS:generated -->
<!-- GENERATED_AT:%s -->
`, personaName, time.Now().Format(time.RFC3339), time.Now().Format(time.RFC3339))

	outputPath := filepath.Join(g.baseDir, "personas", personaName, "chatbot_config.json")
	if err := os.WriteFile(outputPath, []byte(content), 0644); err != nil {
		return fmt.Errorf("failed to write chatbot config file: %w", err)
	}

	g.logger.Infof("Successfully generated chatbot configuration: %s", outputPath)
	return nil
}

// GenerateAPIEndpoint generates API endpoint configuration
func (g *GeneratorService) GenerateAPIEndpoint(ctx context.Context, personaName string, assetType AssetType) error {
	g.logger.Infof("Generating API endpoint configuration for persona: %s", personaName)

	content := fmt.Sprintf(`# %s - API Endpoint Configuration

> **Generated on:** %s  
> **Purpose:** API deployment and endpoint configuration  
> **Source:** synthesized.md  

## API Configuration
This is a placeholder for API endpoint configuration.

### Endpoint Details
- **Base URL:** [Placeholder]
- **Methods:** [Placeholder]
- **Authentication:** [Placeholder]
- **Rate Limiting:** [Placeholder]

### Request/Response Format
- **Content Type:** [Placeholder]
- **Schema:** [Placeholder]
- **Error Handling:** [Placeholder]

---
*This is a placeholder asset. Implement actual API endpoint generation logic in generators.go*

<!-- ASSET_STATUS:generated -->
<!-- GENERATED_AT:%s -->
`, personaName, time.Now().Format(time.RFC3339), time.Now().Format(time.RFC3339))

	outputPath := filepath.Join(g.baseDir, "personas", personaName, "api_endpoint.md")
	if err := os.WriteFile(outputPath, []byte(content), 0644); err != nil {
		return fmt.Errorf("failed to write API endpoint file: %w", err)
	}

	g.logger.Infof("Successfully generated API endpoint configuration: %s", outputPath)
	return nil
}

// GetGeneratorCallback returns the appropriate generator callback for an asset type
func (g *GeneratorService) GetGeneratorCallback(assetType AssetType) AssetCallback {
	switch assetType {
	case AssetTypePromptReady:
		return g.GeneratePromptReady
	case AssetTypePlatformAdapt:
		return g.GeneratePlatformAdaptations
	case AssetTypeVoiceClone:
		return g.GenerateVoiceClone
	case AssetTypeImageAvatar:
		return g.GenerateImageAvatar
	case AssetTypeChatbotConfig:
		return g.GenerateChatbotConfig
	case AssetTypeAPIEndpoint:
		return g.GenerateAPIEndpoint
	default:
		return func(ctx context.Context, personaName string, assetType AssetType) error {
			return fmt.Errorf("unknown asset type: %s", assetType)
		}
	}
}

// RegisterAllCallbacks registers all generator callbacks with a monitor
func (g *GeneratorService) RegisterAllCallbacks(monitor *Monitor) {
	assetTypes := []AssetType{
		AssetTypePromptReady,
		AssetTypePlatformAdapt,
		AssetTypeVoiceClone,
		AssetTypeImageAvatar,
		AssetTypeChatbotConfig,
		AssetTypeAPIEndpoint,
		// Note: Prompt generation assets are registered separately by the prompts.Service
	}

	for _, assetType := range assetTypes {
		monitor.RegisterCallback(assetType, g.GetGeneratorCallback(assetType))
	}

	g.logger.Info("Registered basic asset generator callbacks")
}

// Helper function to truncate content for previews
func truncateContent(content string, maxLength int) string {
	if len(content) <= maxLength {
		return content
	}
	return content[:maxLength] + "..."
}
