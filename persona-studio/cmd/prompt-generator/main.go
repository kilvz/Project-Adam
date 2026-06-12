package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/twin2ai/studio/internal/assets"
	"github.com/twin2ai/studio/internal/config"
	"github.com/twin2ai/studio/internal/gemini"
	"github.com/twin2ai/studio/internal/prompts"
)

func main() {
	var (
		baseDir    = flag.String("base", ".", "Base directory for personas")
		logLevel   = flag.String("log", "info", "Log level (debug, info, warn, error)")
		persona    = flag.String("persona", "", "Specific persona name to generate prompts for")
		runOnce    = flag.Bool("once", false, "Run once instead of continuous monitoring")
		interval   = flag.Duration("interval", 5*time.Minute, "Monitoring interval")
		promptType = flag.String("type", "all", "Type of prompts to generate (all, platform, variation)")
		force      = flag.Bool("force", false, "Force regeneration even if prompts exist")
		stats      = flag.Bool("stats", false, "Show prompt generation statistics")
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

	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		logger.Fatalf("Failed to load config: %v", err)
	}

	// Create Gemini client
	geminiClient := gemini.NewClient(cfg.AI.Gemini.APIKey, cfg.AI.Gemini.Model, logger)

	// Create prompt service
	promptService := prompts.NewService(geminiClient, logger, *baseDir)

	// Create asset monitor
	monitor := assets.NewMonitor(*baseDir, logger)

	// Register prompt generation callbacks
	promptService.RegisterCallbacks(monitor)

	ctx := context.Background()

	if *stats {
		// Show statistics
		err := showStatistics(promptService, *persona, logger)
		if err != nil {
			logger.Errorf("Failed to show statistics: %v", err)
			os.Exit(1)
		}
		return
	}

	if *persona != "" {
		// Generate prompts for specific persona
		err := generateForPersona(ctx, promptService, *persona, *promptType, *force, logger)
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
	err = monitor.StartMonitoring(ctx, *interval)
	if err != nil {
		logger.Errorf("Prompt generation monitoring failed: %v", err)
		os.Exit(1)
	}
}

func generateForPersona(ctx context.Context, service *prompts.Service, personaName, promptType string, force bool, logger *logrus.Logger) error {
	logger.Infof("Generating %s prompts for persona: %s", promptType, personaName)

	// Check if prompts are needed (unless forcing)
	if !force {
		needs, err := service.CheckPersonaNeedsPrompts(personaName)
		if err != nil {
			return fmt.Errorf("failed to check if prompts are needed: %w", err)
		}
		if !needs {
			logger.Infof("Persona %s already has up-to-date prompts", personaName)
			return nil
		}
	}

	// Generate based on type
	switch promptType {
	case "all":
		return service.GenerateAllPrompts(ctx, personaName, assets.AssetTypePrompts)
	case "platform":
		return service.GeneratePlatformPrompts(ctx, personaName, assets.AssetTypePlatformPrompts)
	case "variation":
		return service.GenerateVariationPrompts(ctx, personaName, assets.AssetTypeVariationPrompts)
	default:
		return fmt.Errorf("unknown prompt type: %s (use: all, platform, variation)", promptType)
	}
}

func showStatistics(service *prompts.Service, persona string, logger *logrus.Logger) error {
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

// Additional utility functions can be added here for:
// - Batch prompt generation
// - Prompt validation
// - Cleanup of old prompts
// - Export functionality
