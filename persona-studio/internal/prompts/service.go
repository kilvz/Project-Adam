package prompts

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/sirupsen/logrus"
	"github.com/twin2ai/studio/internal/assets"
	"github.com/twin2ai/studio/internal/gemini"
)

// Service handles prompt generation and integration with the asset system
type Service struct {
	generator    *Generator
	repository   *RepositoryManager
	geminiClient *gemini.Client
	githubClient GitHubPRClient // Optional GitHub client for PR creation
	logger       *logrus.Logger
	baseDir      string
	createPRs    bool // Whether to create PRs for prompt updates
}

// GitHubPRClient interface for GitHub PR operations
type GitHubPRClient interface {
	CreatePromptUpdatePR(ctx context.Context, data interface{}) (interface{}, error)
}

// NewService creates a new prompt generation service without GitHub integration
func NewService(geminiClient *gemini.Client, logger *logrus.Logger, baseDir string) *Service {
	generator := NewGenerator(geminiClient, logger, baseDir)
	repository := NewRepositoryManager(baseDir, logger)

	return &Service{
		generator:    generator,
		repository:   repository,
		geminiClient: geminiClient,
		githubClient: nil,
		logger:       logger,
		baseDir:      baseDir,
		createPRs:    false,
	}
}

// NewServiceWithGitHub creates a new prompt generation service with GitHub PR integration
func NewServiceWithGitHub(geminiClient *gemini.Client, githubClient GitHubPRClient, logger *logrus.Logger, baseDir string) *Service {
	generator := NewGenerator(geminiClient, logger, baseDir)
	repository := NewRepositoryManager(baseDir, logger)

	return &Service{
		generator:    generator,
		repository:   repository,
		geminiClient: geminiClient,
		githubClient: githubClient,
		logger:       logger,
		baseDir:      baseDir,
		createPRs:    true,
	}
}

// GenerateAllPrompts is an asset callback that generates all prompts for a persona
func (s *Service) GenerateAllPrompts(ctx context.Context, personaName string, assetType assets.AssetType) error {
	s.logger.Infof("Generating all prompts for persona: %s (triggered by asset type: %s)", personaName, assetType)

	// Read the synthesized persona content
	synthesizedPath := s.getSynthesizedPath(personaName)
	synthesizedContent, err := os.ReadFile(synthesizedPath)
	if err != nil {
		return fmt.Errorf("failed to read synthesized persona: %w", err)
	}

	// Generate all prompts
	results, err := s.generator.GenerateAllPrompts(ctx, personaName, string(synthesizedContent))
	if err != nil {
		return fmt.Errorf("failed to generate prompts: %w", err)
	}

	// Save results to repository
	if err := s.repository.SavePromptResults(ctx, personaName, results); err != nil {
		return fmt.Errorf("failed to save prompt results: %w", err)
	}

	s.logger.Infof("Successfully generated and saved all prompts for %s", personaName)
	return nil
}

// GeneratePlatformPrompts generates only platform-specific prompts
func (s *Service) GeneratePlatformPrompts(ctx context.Context, personaName string, assetType assets.AssetType) error {
	s.logger.Infof("Generating platform prompts for persona: %s", personaName)

	synthesizedPath := s.getSynthesizedPath(personaName)
	synthesizedContent, err := os.ReadFile(synthesizedPath)
	if err != nil {
		return fmt.Errorf("failed to read synthesized persona: %w", err)
	}

	platformTypes := []PromptType{
		PromptTypeChatGPT,
		PromptTypeClaude,
		PromptTypeGemini,
		PromptTypeDiscord,
		PromptTypeCharacterAI,
	}

	var results []PromptResult
	for _, promptType := range platformTypes {
		result, err := s.generator.GeneratePrompt(ctx, personaName, string(synthesizedContent), promptType)
		if err != nil {
			s.logger.Errorf("Failed to generate %s prompt: %v", promptType, err)
			continue
		}
		results = append(results, *result)
	}

	if err := s.repository.SavePromptResults(ctx, personaName, results); err != nil {
		return fmt.Errorf("failed to save platform prompt results: %w", err)
	}

	s.logger.Infof("Successfully generated %d platform prompts for %s", len(results), personaName)
	return nil
}

// GenerateVariationPrompts generates only variation prompts
func (s *Service) GenerateVariationPrompts(ctx context.Context, personaName string, assetType assets.AssetType) error {
	s.logger.Infof("Generating variation prompts for persona: %s", personaName)

	synthesizedPath := s.getSynthesizedPath(personaName)
	synthesizedContent, err := os.ReadFile(synthesizedPath)
	if err != nil {
		return fmt.Errorf("failed to read synthesized persona: %w", err)
	}

	variationTypes := []PromptType{
		PromptTypeCondensed,
		PromptTypeAlternative,
	}

	var results []PromptResult
	for _, promptType := range variationTypes {
		result, err := s.generator.GeneratePrompt(ctx, personaName, string(synthesizedContent), promptType)
		if err != nil {
			s.logger.Errorf("Failed to generate %s prompt: %v", promptType, err)
			continue
		}
		results = append(results, *result)
	}

	if err := s.repository.SavePromptResults(ctx, personaName, results); err != nil {
		return fmt.Errorf("failed to save variation prompt results: %w", err)
	}

	s.logger.Infof("Successfully generated %d variation prompts for %s", len(results), personaName)
	return nil
}

// RegisterCallbacks registers prompt generation callbacks with the asset monitor
func (s *Service) RegisterCallbacks(monitor *assets.Monitor) {
	// Register callback for all prompts generation
	monitor.RegisterCallback(assets.AssetTypePrompts, s.GenerateAllPrompts)

	// Register callback for platform-specific prompts
	monitor.RegisterCallback(assets.AssetTypePlatformPrompts, s.GeneratePlatformPrompts)

	// Register callback for variation prompts
	monitor.RegisterCallback(assets.AssetTypeVariationPrompts, s.GenerateVariationPrompts)

	s.logger.Info("Registered prompt generation callbacks with asset monitor")
}

// CheckPersonaNeedsPrompts checks if a persona needs prompt generation
func (s *Service) CheckPersonaNeedsPrompts(personaName string) (bool, error) {
	synthesizedPath := s.getSynthesizedPath(personaName)

	// Check if synthesized file exists
	if _, err := os.Stat(synthesizedPath); os.IsNotExist(err) {
		return false, nil
	}

	// Check if prompts directory exists and has content
	promptsDir := filepath.Join(s.getPersonaFolder(personaName), "prompts")
	if _, err := os.Stat(promptsDir); os.IsNotExist(err) {
		return true, nil // No prompts directory means prompts needed
	}

	// Check if synthesized file is newer than prompts
	synthesizedInfo, err := os.Stat(synthesizedPath)
	if err != nil {
		return false, fmt.Errorf("failed to stat synthesized file: %w", err)
	}

	promptsInfo, err := os.Stat(promptsDir)
	if err != nil {
		return true, nil // If we can't stat prompts dir, assume we need to generate
	}

	return synthesizedInfo.ModTime().After(promptsInfo.ModTime()), nil
}

// TriggerPromptGeneration manually triggers prompt generation for a persona
func (s *Service) TriggerPromptGeneration(ctx context.Context, personaName string) error {
	s.logger.Infof("Manually triggering prompt generation for: %s", personaName)

	// Mark prompts as pending in asset status
	statusManager := assets.NewStatusManager(s.baseDir)

	// Mark all prompt types as pending
	promptAssets := []assets.AssetType{
		assets.AssetTypePrompts,
		assets.AssetTypePlatformPrompts,
		assets.AssetTypeVariationPrompts,
	}

	for _, assetType := range promptAssets {
		if err := statusManager.MarkAssetPending(personaName, assetType); err != nil {
			s.logger.Warnf("Failed to mark %s as pending: %v", assetType, err)
		}
	}

	// Generate all prompts
	return s.GenerateAllPrompts(ctx, personaName, assets.AssetTypePrompts)
}

// getSynthesizedPath returns the path to the synthesized.md file for a persona
func (s *Service) getSynthesizedPath(personaName string) string {
	return filepath.Join(s.getPersonaFolder(personaName), "synthesized.md")
}

// getPersonaFolder returns the folder path for a persona
func (s *Service) getPersonaFolder(personaName string) string {
	folderName := s.normalizePersonaName(personaName)
	return filepath.Join(s.baseDir, "personas", folderName)
}

// normalizePersonaName converts persona name to folder-safe format
func (s *Service) normalizePersonaName(name string) string {
	// Convert to lowercase and replace spaces with underscores
	// This should match the logic used in structured_pr.go
	folderName := strings.ToLower(strings.ReplaceAll(name, " ", "_"))
	folderName = strings.ReplaceAll(folderName, "/", "_")
	return folderName
}

// GetPromptGenerationStats returns statistics about prompt generation
func (s *Service) GetPromptGenerationStats(personaName string) (map[string]interface{}, error) {
	promptsDir := filepath.Join(s.getPersonaFolder(personaName), "prompts")

	stats := map[string]interface{}{
		"persona_name":    personaName,
		"prompts_dir":     promptsDir,
		"prompts_exist":   false,
		"prompt_count":    0,
		"platform_count":  0,
		"variation_count": 0,
		"files":           []string{},
	}

	// Check if prompts directory exists
	entries, err := os.ReadDir(promptsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return stats, nil
		}
		return nil, fmt.Errorf("failed to read prompts directory: %w", err)
	}

	stats["prompts_exist"] = true

	var files []string
	platformCount := 0
	variationCount := 0

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		filename := entry.Name()
		files = append(files, filename)

		// Count prompt types
		if strings.Contains(filename, "chatgpt") || strings.Contains(filename, "claude") ||
			strings.Contains(filename, "gemini") || strings.Contains(filename, "discord") ||
			strings.Contains(filename, "characterai") {
			platformCount++
		} else if strings.Contains(filename, "condensed") || strings.Contains(filename, "alternative") {
			variationCount++
		}
	}

	stats["prompt_count"] = len(files)
	stats["platform_count"] = platformCount
	stats["variation_count"] = variationCount
	stats["files"] = files

	return stats, nil
}
