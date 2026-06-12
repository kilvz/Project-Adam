package assets

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// GitHubClient interface for GitHub operations needed by the monitor
type GitHubClient interface {
	ListPersonaFolders(ctx context.Context) ([]string, error)
	GetFileModTime(ctx context.Context, filePath string) (time.Time, error)
	FileExists(ctx context.Context, filePath string) bool
}

// Monitor handles monitoring for asset generation triggers
type Monitor struct {
	statusManager *StatusManager
	logger        *logrus.Logger
	baseDir       string
	callbacks     map[AssetType]AssetCallback
	githubClient  GitHubClient
}

// AssetCallback represents a function that gets called when an asset needs generation
type AssetCallback func(ctx context.Context, personaName string, assetType AssetType) error

// PersonaAssetTrigger represents a detected trigger for asset generation
type PersonaAssetTrigger struct {
	PersonaName   string
	AssetTypes    []AssetType
	TriggerReason string
	DetectedAt    time.Time
}

// NewMonitor creates a new asset monitor
func NewMonitor(baseDir string, logger *logrus.Logger) *Monitor {
	return &Monitor{
		statusManager: NewStatusManager(baseDir),
		logger:        logger,
		baseDir:       baseDir,
		callbacks:     make(map[AssetType]AssetCallback),
		githubClient:  nil, // Will be set via SetGitHubClient
	}
}

// NewMonitorWithGitHub creates a new asset monitor with GitHub integration
func NewMonitorWithGitHub(baseDir string, logger *logrus.Logger, githubClient GitHubClient) *Monitor {
	return &Monitor{
		statusManager: NewStatusManager(baseDir),
		logger:        logger,
		baseDir:       baseDir,
		callbacks:     make(map[AssetType]AssetCallback),
		githubClient:  githubClient,
	}
}

// SetGitHubClient sets the GitHub client for the monitor
func (m *Monitor) SetGitHubClient(client GitHubClient) {
	m.githubClient = client
}

// RegisterCallback registers a callback for a specific asset type
func (m *Monitor) RegisterCallback(assetType AssetType, callback AssetCallback) {
	m.callbacks[assetType] = callback
}

// ScanForTriggers scans all personas for asset generation triggers
func (m *Monitor) ScanForTriggers(ctx context.Context) ([]PersonaAssetTrigger, error) {
	m.logger.Debug("Scanning for asset generation triggers...")

	// Use GitHub if available, otherwise fall back to local filesystem
	if m.githubClient != nil {
		return m.scanGitHubForTriggers(ctx)
	}

	return m.scanLocalForTriggers(ctx)
}

// scanGitHubForTriggers scans GitHub repository for persona triggers
func (m *Monitor) scanGitHubForTriggers(ctx context.Context) ([]PersonaAssetTrigger, error) {
	m.logger.Debug("Scanning GitHub repository for persona triggers...")

	// Get list of persona folders from GitHub
	personaNames, err := m.githubClient.ListPersonaFolders(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list persona folders from GitHub: %w", err)
	}

	var triggers []PersonaAssetTrigger

	for _, personaName := range personaNames {
		trigger, err := m.checkGitHubPersonaForTriggers(ctx, personaName)
		if err != nil {
			m.logger.Warnf("Failed to check GitHub persona %s for triggers: %v", personaName, err)
			continue
		}

		if trigger != nil {
			triggers = append(triggers, *trigger)
		}
	}

	m.logger.Debugf("Found %d asset generation triggers from GitHub", len(triggers))
	return triggers, nil
}

// scanLocalForTriggers scans local filesystem for persona triggers (fallback)
func (m *Monitor) scanLocalForTriggers(ctx context.Context) ([]PersonaAssetTrigger, error) {
	m.logger.Debug("Scanning local filesystem for persona triggers...")

	personasDir := filepath.Join(m.baseDir, "personas")
	if _, err := os.Stat(personasDir); os.IsNotExist(err) {
		m.logger.Debug("Personas directory doesn't exist yet")
		return nil, nil
	}

	entries, err := os.ReadDir(personasDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read personas directory: %w", err)
	}

	var triggers []PersonaAssetTrigger

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		personaName := entry.Name()
		trigger, err := m.checkPersonaForTriggers(personaName)
		if err != nil {
			m.logger.Warnf("Failed to check persona %s for triggers: %v", personaName, err)
			continue
		}

		if trigger != nil {
			triggers = append(triggers, *trigger)
		}
	}

	m.logger.Debugf("Found %d asset generation triggers", len(triggers))
	return triggers, nil
}

// checkGitHubPersonaForTriggers checks a specific persona in GitHub for asset generation triggers
func (m *Monitor) checkGitHubPersonaForTriggers(ctx context.Context, personaName string) (*PersonaAssetTrigger, error) {
	// Normalize persona name for GitHub path
	folderName := strings.ToLower(strings.ReplaceAll(personaName, " ", "_"))
	folderName = strings.ReplaceAll(folderName, "/", "_")

	// Check if synthesized.md exists in GitHub
	synthesizedPath := fmt.Sprintf("personas/%s/synthesized.md", folderName)
	if !m.githubClient.FileExists(ctx, synthesizedPath) {
		m.logger.Debugf("No synthesized.md found for %s, skipping", personaName)
		return nil, nil
	}

	// Get modification time of synthesized.md from GitHub
	synthesizedModTime, err := m.githubClient.GetFileModTime(ctx, synthesizedPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get synthesized.md mod time for %s: %w", personaName, err)
	}

	// Load local status to check if generation is needed
	status, err := m.statusManager.LoadStatus(personaName)
	if err != nil {
		m.logger.Debugf("Failed to load status for %s, assuming first generation: %v", personaName, err)
		// Treat as first generation - create trigger for all prompt assets
		return &PersonaAssetTrigger{
			PersonaName:   personaName,
			AssetTypes:    []AssetType{AssetTypePrompts, AssetTypePlatformPrompts, AssetTypeVariationPrompts},
			TriggerReason: "first generation for persona",
			DetectedAt:    time.Now(),
		}, nil
	}

	// Check if synthesized.md was modified after last asset generation
	var triggerReason string
	var assetTypes []AssetType

	if synthesizedModTime.After(status.LastAssetsGeneration) {
		triggerReason = "synthesized.md file was modified in GitHub"
		// Add all prompt-related assets to regeneration queue
		assetTypes = append(assetTypes, AssetTypePrompts, AssetTypePlatformPrompts, AssetTypeVariationPrompts)
		m.logger.Infof("GitHub synthesized.md for %s modified at %v (last generation: %v)",
			personaName, synthesizedModTime, status.LastAssetsGeneration)
	} else if len(status.PendingAssets) > 0 {
		triggerReason = "pending assets detected"
		for _, asset := range status.PendingAssets {
			assetTypes = append(assetTypes, AssetType(asset))
		}
	}

	if len(assetTypes) == 0 {
		return nil, nil
	}

	return &PersonaAssetTrigger{
		PersonaName:   personaName,
		AssetTypes:    assetTypes,
		TriggerReason: triggerReason,
		DetectedAt:    time.Now(),
	}, nil
}

// checkPersonaForTriggers checks a specific persona for asset generation triggers
func (m *Monitor) checkPersonaForTriggers(personaName string) (*PersonaAssetTrigger, error) {
	// Check if synthesized.md was modified after last asset generation
	modified, err := m.statusManager.CheckSynthesizedFileModified(personaName)
	if err != nil {
		return nil, fmt.Errorf("failed to check synthesized file: %w", err)
	}

	status, err := m.statusManager.LoadStatus(personaName)
	if err != nil {
		return nil, fmt.Errorf("failed to load status: %w", err)
	}

	var triggerReason string
	var assetTypes []AssetType

	if modified {
		triggerReason = "synthesized.md file was modified"
		// Add all pending assets to regeneration queue
		for _, asset := range status.PendingAssets {
			assetTypes = append(assetTypes, AssetType(asset))
		}
	} else if len(status.PendingAssets) > 0 {
		triggerReason = "pending assets detected"
		for _, asset := range status.PendingAssets {
			assetTypes = append(assetTypes, AssetType(asset))
		}
	}

	// Check for trigger markers in README
	readmeTrigger, readmeAssets, err := m.checkReadmeTriggers(personaName)
	if err != nil {
		m.logger.Warnf("Failed to check README triggers for %s: %v", personaName, err)
	} else if readmeTrigger {
		if triggerReason == "" {
			triggerReason = "README trigger markers detected"
		} else {
			triggerReason += " and README trigger markers detected"
		}
		assetTypes = append(assetTypes, readmeAssets...)
	}

	if len(assetTypes) == 0 {
		return nil, nil
	}

	return &PersonaAssetTrigger{
		PersonaName:   personaName,
		AssetTypes:    assetTypes,
		TriggerReason: triggerReason,
		DetectedAt:    time.Now(),
	}, nil
}

// checkReadmeTriggers checks for trigger markers in the persona's README
func (m *Monitor) checkReadmeTriggers(personaName string) (bool, []AssetType, error) {
	readmePath := filepath.Join(m.baseDir, "personas", personaName, "README.md")

	data, err := os.ReadFile(readmePath)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil, nil
		}
		return false, nil, fmt.Errorf("failed to read README: %w", err)
	}

	content := string(data)
	var assetTypes []AssetType

	// Look for trigger markers like: <!-- GENERATE:prompt_ready -->
	triggers := []string{
		"<!-- GENERATE:prompt_ready -->",
		"<!-- GENERATE:platform_adaptations -->",
		"<!-- GENERATE:voice_clone -->",
		"<!-- GENERATE:image_avatar -->",
		"<!-- GENERATE:chatbot_config -->",
		"<!-- GENERATE:api_endpoint -->",
		"<!-- GENERATE:prompts -->",
		"<!-- GENERATE:platform_prompts -->",
		"<!-- GENERATE:variation_prompts -->",
	}

	assetMap := map[string]AssetType{
		"prompt_ready":         AssetTypePromptReady,
		"platform_adaptations": AssetTypePlatformAdapt,
		"voice_clone":          AssetTypeVoiceClone,
		"image_avatar":         AssetTypeImageAvatar,
		"chatbot_config":       AssetTypeChatbotConfig,
		"api_endpoint":         AssetTypeAPIEndpoint,
		"prompts":              AssetTypePrompts,
		"platform_prompts":     AssetTypePlatformPrompts,
		"variation_prompts":    AssetTypeVariationPrompts,
	}

	for _, trigger := range triggers {
		if strings.Contains(content, trigger) {
			// Extract asset type from trigger
			parts := strings.Split(trigger, ":")
			if len(parts) == 2 {
				assetTypeStr := strings.TrimSuffix(parts[1], " -->")
				if assetType, exists := assetMap[assetTypeStr]; exists {
					assetTypes = append(assetTypes, assetType)
				}
			}
		}
	}

	return len(assetTypes) > 0, assetTypes, nil
}

// ProcessTriggers processes detected triggers and calls appropriate callbacks
func (m *Monitor) ProcessTriggers(ctx context.Context, triggers []PersonaAssetTrigger) error {
	for _, trigger := range triggers {
		m.logger.Infof("Processing asset generation trigger for %s: %s",
			trigger.PersonaName, trigger.TriggerReason)

		for _, assetType := range trigger.AssetTypes {
			// Mark asset as pending
			if err := m.statusManager.MarkAssetPending(trigger.PersonaName, assetType); err != nil {
				m.logger.Errorf("Failed to mark asset %s as pending for %s: %v",
					assetType, trigger.PersonaName, err)
				continue
			}

			// Call registered callback if available
			if callback, exists := m.callbacks[assetType]; exists {
				m.logger.Infof("Executing callback for %s asset generation", assetType)
				if err := callback(ctx, trigger.PersonaName, assetType); err != nil {
					m.logger.Errorf("Asset generation callback failed for %s/%s: %v",
						trigger.PersonaName, assetType, err)
					continue
				}

				// Mark as generated on successful callback
				if err := m.statusManager.MarkAssetGenerated(trigger.PersonaName, assetType); err != nil {
					m.logger.Errorf("Failed to mark asset %s as generated for %s: %v",
						assetType, trigger.PersonaName, err)
				}
			} else {
				m.logger.Infof("No callback registered for asset type %s, marked as pending", assetType)
			}
		}
	}

	return nil
}

// RunOnce performs a single scan and processing cycle
func (m *Monitor) RunOnce(ctx context.Context) error {
	triggers, err := m.ScanForTriggers(ctx)
	if err != nil {
		return fmt.Errorf("failed to scan for triggers: %w", err)
	}

	if len(triggers) == 0 {
		m.logger.Debug("No asset generation triggers found")
		return nil
	}

	return m.ProcessTriggers(ctx, triggers)
}

// StartMonitoring starts continuous monitoring (for future use)
func (m *Monitor) StartMonitoring(ctx context.Context, interval time.Duration) error {
	m.logger.Infof("Starting asset generation monitoring with %v interval", interval)

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			m.logger.Info("Asset monitoring stopped")
			return ctx.Err()
		case <-ticker.C:
			if err := m.RunOnce(ctx); err != nil {
				m.logger.Errorf("Asset monitoring cycle failed: %v", err)
			}
		}
	}
}
