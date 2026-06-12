package assets

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// AssetStatus represents the status of asset generation for a persona
type AssetStatus struct {
	PersonaName           string            `json:"persona_name"`
	LastSynthesizedUpdate time.Time         `json:"last_synthesized_update"`
	LastAssetsGeneration  time.Time         `json:"last_assets_generation"`
	PendingAssets         []string          `json:"pending_assets"`
	GeneratedAssets       []string          `json:"generated_assets"`
	AssetGenerationFlags  map[string]bool   `json:"asset_generation_flags"`
	Metadata              map[string]string `json:"metadata,omitempty"`
}

// AssetType represents different types of assets that can be generated
type AssetType string

const (
	AssetTypePromptReady   AssetType = "prompt_ready"
	AssetTypePlatformAdapt AssetType = "platform_adaptations"
	AssetTypeVoiceClone    AssetType = "voice_clone"
	AssetTypeImageAvatar   AssetType = "image_avatar"
	AssetTypeChatbotConfig AssetType = "chatbot_config"
	AssetTypeAPIEndpoint   AssetType = "api_endpoint"

	// Prompt generation assets
	AssetTypePrompts          AssetType = "prompts"
	AssetTypePlatformPrompts  AssetType = "platform_prompts"
	AssetTypeVariationPrompts AssetType = "variation_prompts"
)

// StatusManager handles asset status tracking for personas
type StatusManager struct {
	baseDir string
}

// NewStatusManager creates a new status manager
func NewStatusManager(baseDir string) *StatusManager {
	return &StatusManager{
		baseDir: baseDir,
	}
}

// GetStatusFilePath returns the path to the asset status file for a persona
func (sm *StatusManager) GetStatusFilePath(personaName string) string {
	folderName := normalizePersonaName(personaName)
	return filepath.Join(sm.baseDir, "personas", folderName, ".assets_status.json")
}

// LoadStatus loads the asset status for a persona
func (sm *StatusManager) LoadStatus(personaName string) (*AssetStatus, error) {
	statusFile := sm.GetStatusFilePath(personaName)

	data, err := os.ReadFile(statusFile)
	if err != nil {
		if os.IsNotExist(err) {
			// Return default status if file doesn't exist
			return sm.createDefaultStatus(personaName), nil
		}
		return nil, fmt.Errorf("failed to read status file: %w", err)
	}

	var status AssetStatus
	if err := json.Unmarshal(data, &status); err != nil {
		return nil, fmt.Errorf("failed to parse status file: %w", err)
	}

	return &status, nil
}

// SaveStatus saves the asset status for a persona
func (sm *StatusManager) SaveStatus(status *AssetStatus) error {
	statusFile := sm.GetStatusFilePath(status.PersonaName)

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(statusFile), 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	data, err := json.MarshalIndent(status, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal status: %w", err)
	}

	if err := os.WriteFile(statusFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write status file: %w", err)
	}

	return nil
}

// UpdateSynthesizedTimestamp updates the last synthesized update timestamp
func (sm *StatusManager) UpdateSynthesizedTimestamp(personaName string) error {
	status, err := sm.LoadStatus(personaName)
	if err != nil {
		return fmt.Errorf("failed to load status: %w", err)
	}

	status.LastSynthesizedUpdate = time.Now()

	return sm.SaveStatus(status)
}

// MarkAssetPending marks an asset as pending generation
func (sm *StatusManager) MarkAssetPending(personaName string, assetType AssetType) error {
	status, err := sm.LoadStatus(personaName)
	if err != nil {
		return fmt.Errorf("failed to load status: %w", err)
	}

	assetStr := string(assetType)

	// Add to pending if not already there
	if !contains(status.PendingAssets, assetStr) {
		status.PendingAssets = append(status.PendingAssets, assetStr)
	}

	// Remove from generated if it was there
	status.GeneratedAssets = removeFromSlice(status.GeneratedAssets, assetStr)

	// Set flag
	if status.AssetGenerationFlags == nil {
		status.AssetGenerationFlags = make(map[string]bool)
	}
	status.AssetGenerationFlags[assetStr] = false

	return sm.SaveStatus(status)
}

// MarkAssetGenerated marks an asset as successfully generated
func (sm *StatusManager) MarkAssetGenerated(personaName string, assetType AssetType) error {
	status, err := sm.LoadStatus(personaName)
	if err != nil {
		return fmt.Errorf("failed to load status: %w", err)
	}

	assetStr := string(assetType)

	// Remove from pending
	status.PendingAssets = removeFromSlice(status.PendingAssets, assetStr)

	// Add to generated if not already there
	if !contains(status.GeneratedAssets, assetStr) {
		status.GeneratedAssets = append(status.GeneratedAssets, assetStr)
	}

	// Set flag and timestamp
	if status.AssetGenerationFlags == nil {
		status.AssetGenerationFlags = make(map[string]bool)
	}
	status.AssetGenerationFlags[assetStr] = true
	status.LastAssetsGeneration = time.Now()

	return sm.SaveStatus(status)
}

// NeedsAssetGeneration checks if any assets need to be generated
func (sm *StatusManager) NeedsAssetGeneration(personaName string) (bool, []AssetType, error) {
	status, err := sm.LoadStatus(personaName)
	if err != nil {
		return false, nil, fmt.Errorf("failed to load status: %w", err)
	}

	// Check if synthesized was updated after last asset generation
	if status.LastSynthesizedUpdate.After(status.LastAssetsGeneration) {
		var pendingTypes []AssetType
		for _, asset := range status.PendingAssets {
			pendingTypes = append(pendingTypes, AssetType(asset))
		}
		return true, pendingTypes, nil
	}

	return len(status.PendingAssets) > 0, nil, nil
}

// GetSynthesizedFilePath returns the path to the synthesized.md file
func (sm *StatusManager) GetSynthesizedFilePath(personaName string) string {
	folderName := normalizePersonaName(personaName)
	return filepath.Join(sm.baseDir, "personas", folderName, "synthesized.md")
}

// CheckSynthesizedFileModified checks if synthesized.md was modified after last asset generation
func (sm *StatusManager) CheckSynthesizedFileModified(personaName string) (bool, error) {
	status, err := sm.LoadStatus(personaName)
	if err != nil {
		return false, fmt.Errorf("failed to load status: %w", err)
	}

	synthesizedFile := sm.GetSynthesizedFilePath(personaName)
	fileInfo, err := os.Stat(synthesizedFile)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil // File doesn't exist yet
		}
		return false, fmt.Errorf("failed to stat synthesized file: %w", err)
	}

	// Compare file modification time with last asset generation
	return fileInfo.ModTime().After(status.LastAssetsGeneration), nil
}

// createDefaultStatus creates a default status for a new persona
func (sm *StatusManager) createDefaultStatus(personaName string) *AssetStatus {
	return &AssetStatus{
		PersonaName:           personaName,
		LastSynthesizedUpdate: time.Now(),
		LastAssetsGeneration:  time.Time{}, // Zero time means never generated
		PendingAssets:         []string{},
		GeneratedAssets:       []string{},
		AssetGenerationFlags:  make(map[string]bool),
		Metadata:              make(map[string]string),
	}
}

// normalizePersonaName converts persona name to folder-safe format
func normalizePersonaName(name string) string {
	// This should match the logic in structured_pr.go
	folderName := name
	folderName = filepath.Clean(folderName)
	// Add any additional normalization logic here
	return folderName
}

// SerializeAssetStatus converts an AssetStatus to formatted JSON string
func SerializeAssetStatus(status *AssetStatus) (string, error) {
	data, err := json.MarshalIndent(status, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal asset status: %w", err)
	}
	return string(data), nil
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
