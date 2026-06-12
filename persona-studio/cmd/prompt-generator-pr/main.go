package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/sirupsen/logrus"
	"github.com/twin2ai/studio/internal/assets"
	"github.com/twin2ai/studio/internal/config"
	"github.com/twin2ai/studio/internal/gemini"
	"github.com/twin2ai/studio/internal/github"
	"github.com/twin2ai/studio/internal/prompts"
)

func main() {
	// Load .env file if it exists
	if err := godotenv.Load(); err != nil {
		// Don't fail if .env doesn't exist, just continue with environment variables
		// fmt.Printf("Note: .env file not found, using environment variables only\n")
	}

	var (
		baseDir     = flag.String("base", ".", "Base directory for personas")
		logLevel    = flag.String("log", "info", "Log level (debug, info, warn, error)")
		persona     = flag.String("persona", "", "Specific persona name to generate prompts for")
		runOnce     = flag.Bool("once", false, "Run once instead of continuous monitoring")
		interval    = flag.Duration("interval", 5*time.Minute, "Monitoring interval")
		promptType  = flag.String("type", "all", "Type of prompts to generate (all, platform, variation)")
		force       = flag.Bool("force", false, "Force regeneration even if prompts exist")
		stats       = flag.Bool("stats", false, "Show prompt generation statistics")
		createPR    = flag.Bool("pr", true, "Create GitHub pull request with generated prompts")
		issueNumber = flag.Int("issue", 0, "Optional issue number to link the PR to")
	)
	flag.Parse()

	// Setup logger
	logger := logrus.New()
	level, err := logrus.ParseLevel(*logLevel)
	if err != nil {
		fmt.Printf("Invalid log level: %v\n", err)
		os.Exit(1)
	}
	logger.SetLevel(level)

	// Load configuration from .env file
	cfg, err := config.Load()
	if err != nil {
		logger.Fatalf("Failed to load config: %v", err)
	}

	// Validate required configuration for PR creation
	if *createPR {
		if cfg.GitHub.Token == "" {
			logger.Fatalf("GitHub token required for PR creation. Please set GITHUB_TOKEN in your .env file or environment.")
		}
		if cfg.AI.Gemini.APIKey == "" {
			logger.Fatalf("Gemini API key required for prompt generation. Please set GOOGLE_API_KEY in your .env file or environment.")
		}
	}

	// Create Gemini client
	geminiClient := gemini.NewClient(cfg.AI.Gemini.APIKey, cfg.AI.Gemini.Model, logger)

	// Create GitHub client if PR creation is enabled
	var githubService *prompts.GitHubService
	var monitor *assets.Monitor

	if *createPR {
		githubClient := github.NewClient(
			cfg.GitHub.Token,
			cfg.GitHub.Owner,
			cfg.GitHub.Repo,
			cfg.GitHub.PersonasOwner,
			cfg.GitHub.PersonasRepo,
			cfg.GitHub.PersonaLabel,
			logger,
		)

		githubService = prompts.NewGitHubService(geminiClient, githubClient, logger, *baseDir)
		monitor = assets.NewMonitor(*baseDir, logger)
		githubService.RegisterCallbacks(monitor)
	} else {
		// Use regular prompt service without GitHub integration
		promptService := prompts.NewService(geminiClient, logger, *baseDir)
		monitor = assets.NewMonitor(*baseDir, logger)
		promptService.RegisterCallbacks(monitor)
	}

	ctx := context.Background()

	if *stats {
		// Show statistics
		err := showStatistics(githubService, *persona, logger)
		if err != nil {
			logger.Errorf("Failed to show statistics: %v", err)
			os.Exit(1)
		}
		return
	}

	if *persona != "" {
		// Generate prompts for specific persona
		var err error
		if *createPR && githubService != nil {
			err = generateForPersonaWithPR(ctx, githubService, *persona, *promptType, *force, *issueNumber, logger)
		} else {
			err = generateForPersona(ctx, githubService, *persona, *promptType, *force, logger)
		}
		if err != nil {
			logger.Errorf("Failed to generate prompts for %s: %v", *persona, err)
			os.Exit(1)
		}
		return
	}

	if *runOnce {
		// Run prompt generation scan once
		logger.Info("Running prompt generation scan once...")
		err := monitor.RunOnce(ctx)
		if err != nil {
			logger.Errorf("Prompt generation scan failed: %v", err)
			os.Exit(1)
		}
		logger.Info("Prompt generation scan completed")
		return
	}

	// Continuous monitoring for prompt generation
	logger.Infof("Starting continuous prompt generation monitoring with %v interval", *interval)
	if *createPR {
		logger.Info("PR creation enabled - prompts will be submitted as pull requests")
	} else {
		logger.Info("PR creation disabled - prompts will be saved locally only")
	}

	err = monitor.StartMonitoring(ctx, *interval)
	if err != nil {
		logger.Errorf("Prompt generation monitoring failed: %v", err)
		os.Exit(1)
	}
}

func generateForPersonaWithPR(ctx context.Context, service *prompts.GitHubService, personaName, promptType string, force bool, issueNumber int, logger *logrus.Logger) error {
	logger.Infof("Generating %s prompts with PR creation for persona: %s", promptType, personaName)

	// Check if prompts are needed (unless forcing)
	if !force {
		needs, err := service.GetPromptGenerationStats(personaName)
		if err != nil {
			return fmt.Errorf("failed to check if prompts are needed: %w", err)
		}
		if promptsExist, ok := needs["prompts_exist"].(bool); ok && promptsExist {
			logger.Infof("Persona %s already has prompts", personaName)
			if !force {
				return nil
			}
		}
	}

	// Set issue number if provided
	var issuePtr *int
	if issueNumber > 0 {
		issuePtr = &issueNumber
	}

	// Generate based on type with PR creation
	if force {
		switch promptType {
		case "all":
			return service.TriggerPromptGenerationWithPRForced(ctx, personaName, issuePtr)
		case "platform":
			return service.GeneratePlatformPromptsWithPRForced(ctx, personaName, assets.AssetTypePlatformPrompts)
		case "variation":
			return service.GenerateVariationPromptsWithPRForced(ctx, personaName, assets.AssetTypeVariationPrompts)
		default:
			return fmt.Errorf("unknown prompt type: %s (use: all, platform, variation)", promptType)
		}
	} else {
		switch promptType {
		case "all":
			return service.TriggerPromptGenerationWithPR(ctx, personaName, issuePtr)
		case "platform":
			return service.GeneratePlatformPromptsWithPR(ctx, personaName, assets.AssetTypePlatformPrompts)
		case "variation":
			return service.GenerateVariationPromptsWithPR(ctx, personaName, assets.AssetTypeVariationPrompts)
		default:
			return fmt.Errorf("unknown prompt type: %s (use: all, platform, variation)", promptType)
		}
	}
}

func generateForPersona(ctx context.Context, service *prompts.GitHubService, personaName, promptType string, force bool, logger *logrus.Logger) error {
	logger.Infof("Generating %s prompts locally for persona: %s", promptType, personaName)

	// For non-PR generation, fall back to base service functionality
	// This is a simplified version - you could implement local-only generation here
	logger.Warn("Local-only prompt generation not fully implemented - use -pr flag for full functionality")
	return fmt.Errorf("local-only generation not implemented, use -pr flag")
}

func showStatistics(service *prompts.GitHubService, persona string, logger *logrus.Logger) error {
	if persona != "" {
		// Show stats for specific persona
		stats, err := service.GetPromptGenerationStats(persona)
		if err != nil {
			return fmt.Errorf("failed to get stats for %s: %w", persona, err)
		}

		logger.Infof("Prompt Generation Statistics for %s:", persona)
		for key, value := range stats {
			logger.Infof("  %s: %v", key, value)
		}
		return nil
	}

	// Show stats for all personas
	logger.Info("Showing prompt generation statistics for all personas...")

	// This would require iterating through all personas
	// For now, just show a message
	logger.Info("Global statistics not yet implemented. Use -persona flag for specific persona stats.")

	return nil
}
