package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/twin2ai/studio/internal/assets"
)

func main() {
	var (
		baseDir  = flag.String("base", ".", "Base directory for personas")
		logLevel = flag.String("log", "info", "Log level (debug, info, warn, error)")
		persona  = flag.String("persona", "", "Specific persona name to check")
		runOnce  = flag.Bool("once", false, "Run once instead of continuous monitoring")
		interval = flag.Duration("interval", 30*time.Second, "Monitoring interval")
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

	// Create monitor and generator service
	monitor := assets.NewMonitor(*baseDir, logger)
	generator := assets.NewGeneratorService(*baseDir, logger)

	// Register all callbacks
	generator.RegisterAllCallbacks(monitor)

	ctx := context.Background()

	if *persona != "" {
		// Check specific persona
		err := checkSpecificPersona(ctx, monitor, *persona, logger)
		if err != nil {
			logger.Errorf("Failed to check persona %s: %v", *persona, err)
			os.Exit(1)
		}
		return
	}

	if *runOnce {
		// Run once
		logger.Info("Running asset monitoring scan once...")
		err := monitor.RunOnce(ctx)
		if err != nil {
			logger.Errorf("Monitoring scan failed: %v", err)
			os.Exit(1)
		}
		logger.Info("Asset monitoring scan completed")
		return
	}

	// Continuous monitoring
	logger.Infof("Starting continuous asset monitoring with %v interval", *interval)
	err = monitor.StartMonitoring(ctx, *interval)
	if err != nil {
		logger.Errorf("Monitoring failed: %v", err)
		os.Exit(1)
	}
}

func checkSpecificPersona(ctx context.Context, monitor *assets.Monitor, personaName string, logger *logrus.Logger) error {
	logger.Infof("Checking persona: %s", personaName)

	// Check for asset generation needs
	statusManager := assets.NewStatusManager(".")

	needs, assetTypes, err := statusManager.NeedsAssetGeneration(personaName)
	if err != nil {
		return fmt.Errorf("failed to check asset generation needs: %w", err)
	}

	if needs {
		logger.Infof("Persona %s needs asset generation for: %v", personaName, assetTypes)
	} else {
		logger.Infof("Persona %s does not need asset generation", personaName)
	}

	// Check if synthesized file was modified
	modified, err := statusManager.CheckSynthesizedFileModified(personaName)
	if err != nil {
		return fmt.Errorf("failed to check synthesized file: %w", err)
	}

	if modified {
		logger.Infof("Synthesized file for %s was modified after last asset generation", personaName)
	} else {
		logger.Infof("Synthesized file for %s is up to date", personaName)
	}

	// Load and display current status
	status, err := statusManager.LoadStatus(personaName)
	if err != nil {
		return fmt.Errorf("failed to load status: %w", err)
	}

	logger.Infof("Current status for %s:", personaName)
	logger.Infof("  Last synthesized update: %s", status.LastSynthesizedUpdate.Format(time.RFC3339))
	logger.Infof("  Last assets generation: %s", status.LastAssetsGeneration.Format(time.RFC3339))
	logger.Infof("  Pending assets: %v", status.PendingAssets)
	logger.Infof("  Generated assets: %v", status.GeneratedAssets)

	return nil
}
