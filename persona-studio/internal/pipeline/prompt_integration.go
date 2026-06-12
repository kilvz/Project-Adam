package pipeline

import (
	"context"

	"github.com/sirupsen/logrus"
	"github.com/twin2ai/studio/internal/assets"
	"github.com/twin2ai/studio/internal/gemini"
	"github.com/twin2ai/studio/internal/github"
	"github.com/twin2ai/studio/internal/prompts"
)

// PromptPipelineIntegration handles integration of prompt generation with the main pipeline
type PromptPipelineIntegration struct {
	githubService *prompts.GitHubService
	monitor       *assets.Monitor
	logger        *logrus.Logger
	enabled       bool
}

// NewPromptPipelineIntegration creates a new prompt pipeline integration
func NewPromptPipelineIntegration(geminiClient *gemini.Client, githubClient *github.Client, logger *logrus.Logger, baseDir string, enabled bool) *PromptPipelineIntegration {
	if !enabled {
		return &PromptPipelineIntegration{
			enabled: false,
			logger:  logger,
		}
	}

	githubService := prompts.NewGitHubService(geminiClient, githubClient, logger, baseDir)

	// Create monitor with GitHub integration for repository-wide monitoring
	monitor := assets.NewMonitorWithGitHub(baseDir, logger, githubClient)

	// Register prompt generation callbacks
	githubService.RegisterCallbacks(monitor)

	return &PromptPipelineIntegration{
		githubService: githubService,
		monitor:       monitor,
		logger:        logger,
		enabled:       true,
	}
}

// ProcessPromptGeneration checks for and processes prompt generation triggers
func (ppi *PromptPipelineIntegration) ProcessPromptGeneration(ctx context.Context) error {
	if !ppi.enabled {
		ppi.logger.Debug("Prompt generation integration disabled")
		return nil
	}

	ppi.logger.Debug("Checking for prompt generation triggers...")

	// Run asset monitoring to detect prompt generation needs
	return ppi.monitor.RunOnce(ctx)
}

// TriggerPromptGeneration manually triggers prompt generation for a persona
func (ppi *PromptPipelineIntegration) TriggerPromptGeneration(ctx context.Context, personaName string, issueNumber *int) error {
	if !ppi.enabled {
		ppi.logger.Warn("Prompt generation integration disabled, cannot trigger generation")
		return nil
	}

	ppi.logger.Infof("Manually triggering prompt generation for persona: %s", personaName)
	return ppi.githubService.TriggerPromptGenerationWithPR(ctx, personaName, issueNumber)
}

// IsEnabled returns whether prompt generation integration is enabled
func (ppi *PromptPipelineIntegration) IsEnabled() bool {
	return ppi.enabled
}

// GetStats returns prompt generation statistics for a persona
func (ppi *PromptPipelineIntegration) GetStats(personaName string) (map[string]interface{}, error) {
	if !ppi.enabled {
		return map[string]interface{}{
			"enabled": false,
			"message": "Prompt generation integration disabled",
		}, nil
	}

	return ppi.githubService.GetPromptGenerationStats(personaName)
}
