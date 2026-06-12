package prompts

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/twin2ai/studio/internal/gemini"
)

// PromptType represents different types of prompts that can be generated
type PromptType string

const (
	// Platform-specific prompts
	PromptTypeChatGPT     PromptType = "chatgpt"
	PromptTypeClaude      PromptType = "claude"
	PromptTypeGemini      PromptType = "gemini"
	PromptTypeDiscord     PromptType = "discord"
	PromptTypeCharacterAI PromptType = "characterai"

	// Variation prompts
	PromptTypeCondensed   PromptType = "condensed"
	PromptTypeAlternative PromptType = "alternative"
)

// PromptResult represents the result of prompt generation
type PromptResult struct {
	PromptType  PromptType
	Content     string
	GeneratedAt time.Time
	PersonaName string
	Error       error
}

// Generator handles prompt generation using Gemini Flash
type Generator struct {
	geminiClient *gemini.Client
	logger       *logrus.Logger
	baseDir      string
}

// NewGenerator creates a new prompt generator
func NewGenerator(geminiClient *gemini.Client, logger *logrus.Logger, baseDir string) *Generator {
	return &Generator{
		geminiClient: geminiClient,
		logger:       logger,
		baseDir:      baseDir,
	}
}

// GenerateAllPrompts generates all prompt types for a persona
func (g *Generator) GenerateAllPrompts(ctx context.Context, personaName string, synthesizedContent string) ([]PromptResult, error) {
	g.logger.Infof("Generating all prompts for persona: %s", personaName)

	promptTypes := []PromptType{
		PromptTypeChatGPT,
		PromptTypeClaude,
		PromptTypeGemini,
		PromptTypeDiscord,
		PromptTypeCharacterAI,
		PromptTypeCondensed,
		PromptTypeAlternative,
	}

	var results []PromptResult

	for _, promptType := range promptTypes {
		result, err := g.GeneratePrompt(ctx, personaName, synthesizedContent, promptType)
		if err != nil {
			g.logger.Errorf("Failed to generate %s prompt for %s: %v", promptType, personaName, err)
			results = append(results, PromptResult{
				PromptType:  promptType,
				PersonaName: personaName,
				GeneratedAt: time.Now(),
				Error:       err,
			})
			continue
		}
		results = append(results, *result)
	}

	g.logger.Infof("Generated %d prompts for %s", len(results), personaName)
	return results, nil
}

// GeneratePrompt generates a specific prompt type for a persona
func (g *Generator) GeneratePrompt(ctx context.Context, personaName string, synthesizedContent string, promptType PromptType) (*PromptResult, error) {
	g.logger.Debugf("Generating %s prompt for %s", promptType, personaName)

	// Load the prompt template
	template, err := g.loadPromptTemplate(promptType)
	if err != nil {
		return nil, fmt.Errorf("failed to load prompt template for %s: %w", promptType, err)
	}

	// Replace placeholder with synthesized content
	promptInput := strings.ReplaceAll(template, "{{SYNTHESIZED_PERSONA}}", synthesizedContent)

	// Generate using Gemini Flash
	content, err := g.geminiClient.GeneratePersonaPrompt(ctx, promptInput)
	if err != nil {
		return nil, fmt.Errorf("failed to generate prompt with Gemini: %w", err)
	}

	return &PromptResult{
		PromptType:  promptType,
		Content:     content,
		GeneratedAt: time.Now(),
		PersonaName: personaName,
		Error:       nil,
	}, nil
}

// loadPromptTemplate loads a prompt template from file
func (g *Generator) loadPromptTemplate(promptType PromptType) (string, error) {
	var filename string

	switch promptType {
	case PromptTypeChatGPT:
		filename = "platform_chatgpt.txt"
	case PromptTypeClaude:
		filename = "platform_claude.txt"
	case PromptTypeGemini:
		filename = "platform_gemini.txt"
	case PromptTypeDiscord:
		filename = "platform_discord.txt"
	case PromptTypeCharacterAI:
		filename = "platform_characterai.txt"
	case PromptTypeCondensed:
		filename = "variation_condensed.txt"
	case PromptTypeAlternative:
		filename = "variation_alternative.txt"
	default:
		return "", fmt.Errorf("unknown prompt type: %s", promptType)
	}

	templatePath := filepath.Join(g.baseDir, "prompts", filename)
	data, err := os.ReadFile(templatePath)
	if err != nil {
		return "", fmt.Errorf("failed to read template file %s: %w", templatePath, err)
	}

	return string(data), nil
}

// GetPromptFilename returns the output filename for a prompt type
func GetPromptFilename(promptType PromptType) string {
	switch promptType {
	case PromptTypeChatGPT:
		return "prompts/chatgpt.md"
	case PromptTypeClaude:
		return "prompts/claude.md"
	case PromptTypeGemini:
		return "prompts/gemini.md"
	case PromptTypeDiscord:
		return "prompts/discord.md"
	case PromptTypeCharacterAI:
		return "prompts/characterai.md"
	case PromptTypeCondensed:
		return "prompts/condensed.md"
	case PromptTypeAlternative:
		return "prompts/alternative.md"
	default:
		return fmt.Sprintf("prompts/%s.md", promptType)
	}
}

// GetPromptDisplayName returns a human-readable name for a prompt type
func GetPromptDisplayName(promptType PromptType) string {
	switch promptType {
	case PromptTypeChatGPT:
		return "ChatGPT System Prompt"
	case PromptTypeClaude:
		return "Claude System Prompt"
	case PromptTypeGemini:
		return "Gemini System Prompt"
	case PromptTypeDiscord:
		return "Discord Bot Personality"
	case PromptTypeCharacterAI:
		return "Character.AI Character"
	case PromptTypeCondensed:
		return "Condensed Prompt-Ready Version"
	case PromptTypeAlternative:
		return "Alternative Variations"
	default:
		return string(promptType)
	}
}

// ValidatePromptResult validates a generated prompt result
func ValidatePromptResult(result *PromptResult) error {
	if result.Error != nil {
		return result.Error
	}

	if strings.TrimSpace(result.Content) == "" {
		return fmt.Errorf("generated prompt content is empty")
	}

	// Check minimum length requirements
	minLength := 100
	if len(result.Content) < minLength {
		return fmt.Errorf("generated prompt is too short (minimum %d characters)", minLength)
	}

	// Check for template placeholders that weren't replaced
	if strings.Contains(result.Content, "{{") || strings.Contains(result.Content, "}}") {
		return fmt.Errorf("generated prompt contains unreplaced template placeholders")
	}

	return nil
}

// GetAllPromptTypes returns all available prompt types
func GetAllPromptTypes() []PromptType {
	return []PromptType{
		PromptTypeChatGPT,
		PromptTypeClaude,
		PromptTypeGemini,
		PromptTypeDiscord,
		PromptTypeCharacterAI,
		PromptTypeCondensed,
		PromptTypeAlternative,
	}
}

// IsPlatformPrompt returns true if the prompt type is platform-specific
func IsPlatformPrompt(promptType PromptType) bool {
	switch promptType {
	case PromptTypeChatGPT, PromptTypeClaude, PromptTypeGemini, PromptTypeDiscord, PromptTypeCharacterAI:
		return true
	default:
		return false
	}
}

// IsVariationPrompt returns true if the prompt type is a variation
func IsVariationPrompt(promptType PromptType) bool {
	switch promptType {
	case PromptTypeCondensed, PromptTypeAlternative:
		return true
	default:
		return false
	}
}
