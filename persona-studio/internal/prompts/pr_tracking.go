package prompts

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// PRTracker manages tracking of pending prompt generation PRs
type PRTracker struct {
	baseDir string
	logger  *logrus.Logger
}

// PRRecord represents a tracked PR for prompt generation
type PRRecord struct {
	PersonaName     string
	PRNumber        int
	PRUrl           string
	CreatedAt       time.Time
	SynthesizedHash string // Hash of synthesized content when PR was created
}

// NewPRTracker creates a new PR tracker
func NewPRTracker(baseDir string, logger *logrus.Logger) *PRTracker {
	return &PRTracker{
		baseDir: baseDir,
		logger:  logger,
	}
}

// HasPendingPR checks if there's already a pending PR for this persona
func (pt *PRTracker) HasPendingPR(ctx context.Context, personaName string, githubClient interface{}) (bool, *PRRecord, error) {
	// Load existing PR records
	records, err := pt.loadPRRecords()
	if err != nil {
		pt.logger.Warnf("Failed to load PR records: %v", err)
		records = make(map[string]*PRRecord) // Continue with empty records
	}

	// Check if we have a record for this persona
	record, exists := records[personaName]
	if !exists {
		return false, nil, nil
	}

	// Check if the PR is still open/pending via GitHub API
	if ghClient, ok := githubClient.(interface {
		GetPRStatus(ctx context.Context, prNumber int) (string, error)
	}); ok {
		status, err := ghClient.GetPRStatus(ctx, record.PRNumber)
		if err != nil {
			pt.logger.Warnf("Failed to check PR #%d status, assuming still pending: %v", record.PRNumber, err)
			return true, record, nil
		}

		// If PR is closed or merged, remove from tracking
		if status == "closed" || status == "merged" {
			pt.logger.Infof("PR #%d for %s is %s, removing from tracking", record.PRNumber, personaName, status)
			if err := pt.RemovePR(personaName); err != nil {
				pt.logger.Warnf("Failed to remove closed PR record: %v", err)
			}
			return false, nil, nil
		}

		pt.logger.Debugf("PR #%d for %s is still %s", record.PRNumber, personaName, status)
	} else {
		pt.logger.Debugf("GitHub client doesn't support PR status checking, assuming PR #%d is pending", record.PRNumber)
	}

	return true, record, nil
}

// TrackPR records a new PR for tracking
func (pt *PRTracker) TrackPR(personaName string, prNumber int, prUrl string, synthesizedHash string) error {
	// Load existing records
	records, err := pt.loadPRRecords()
	if err != nil {
		pt.logger.Warnf("Failed to load existing PR records: %v", err)
		records = make(map[string]*PRRecord)
	}

	// Add new record
	records[personaName] = &PRRecord{
		PersonaName:     personaName,
		PRNumber:        prNumber,
		PRUrl:           prUrl,
		CreatedAt:       time.Now(),
		SynthesizedHash: synthesizedHash,
	}

	// Save updated records
	return pt.savePRRecords(records)
}

// RemovePR removes a PR from tracking (when merged/closed)
func (pt *PRTracker) RemovePR(personaName string) error {
	// Load existing records
	records, err := pt.loadPRRecords()
	if err != nil {
		return fmt.Errorf("failed to load PR records: %w", err)
	}

	// Remove the record
	delete(records, personaName)

	// Save updated records
	return pt.savePRRecords(records)
}

// GetContentHash creates a hash of the synthesized content for change detection
func (pt *PRTracker) GetContentHash(content string) string {
	// Simple hash - could use crypto/sha256 for better hashing
	// For now, use length + first/last chars as a simple hash
	if len(content) == 0 {
		return "empty"
	}

	hash := fmt.Sprintf("%d-%c-%c", len(content), content[0], content[len(content)-1])
	return hash
}

// ShouldCreatePR determines if a new PR should be created
func (pt *PRTracker) ShouldCreatePR(ctx context.Context, personaName string, synthesizedContent string, githubClient interface{}) (bool, string, error) {
	// Get hash of current content
	currentHash := pt.GetContentHash(synthesizedContent)

	// Check if there's a pending PR
	hasPending, record, err := pt.HasPendingPR(ctx, personaName, githubClient)
	if err != nil {
		return false, "", fmt.Errorf("failed to check pending PRs: %w", err)
	}

	if !hasPending {
		pt.logger.Debugf("No pending PR found for %s, creating new PR", personaName)
		return true, "no_pending_pr", nil
	}

	// Check if content has changed since the pending PR
	if record.SynthesizedHash != currentHash {
		pt.logger.Infof("Content changed for %s since PR #%d, will create new PR", personaName, record.PRNumber)
		// Remove old record and allow new PR
		if err := pt.RemovePR(personaName); err != nil {
			pt.logger.Warnf("Failed to remove old PR record: %v", err)
		}
		return true, "content_changed", nil
	}

	// Content hasn't changed and PR is pending
	pt.logger.Infof("PR #%d already pending for %s with same content, skipping", record.PRNumber, personaName)
	return false, fmt.Sprintf("pending_pr_%d", record.PRNumber), nil
}

// CleanupOldRecords removes records for PRs older than a certain age
func (pt *PRTracker) CleanupOldRecords(maxAge time.Duration) error {
	records, err := pt.loadPRRecords()
	if err != nil {
		return fmt.Errorf("failed to load PR records: %w", err)
	}

	cutoff := time.Now().Add(-maxAge)
	cleaned := false

	for personaName, record := range records {
		if record.CreatedAt.Before(cutoff) {
			pt.logger.Infof("Removing old PR record for %s (PR #%d, age: %v)",
				personaName, record.PRNumber, time.Since(record.CreatedAt))
			delete(records, personaName)
			cleaned = true
		}
	}

	if cleaned {
		return pt.savePRRecords(records)
	}

	return nil
}

// loadPRRecords loads PR records from file
func (pt *PRTracker) loadPRRecords() (map[string]*PRRecord, error) {
	recordsFile := pt.getPRRecordsFile()

	data, err := os.ReadFile(recordsFile)
	if err != nil {
		if os.IsNotExist(err) {
			return make(map[string]*PRRecord), nil
		}
		return nil, fmt.Errorf("failed to read PR records file: %w", err)
	}

	records := make(map[string]*PRRecord)
	lines := strings.Split(string(data), "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		parts := strings.Split(line, "\t")
		if len(parts) != 5 {
			pt.logger.Warnf("Skipping invalid PR record line: %s", line)
			continue
		}

		prNumber, err := strconv.Atoi(parts[1])
		if err != nil {
			pt.logger.Warnf("Invalid PR number in record: %s", parts[1])
			continue
		}

		createdAt, err := time.Parse(time.RFC3339, parts[3])
		if err != nil {
			pt.logger.Warnf("Invalid timestamp in record: %s", parts[3])
			continue
		}

		records[parts[0]] = &PRRecord{
			PersonaName:     parts[0],
			PRNumber:        prNumber,
			PRUrl:           parts[2],
			CreatedAt:       createdAt,
			SynthesizedHash: parts[4],
		}
	}

	return records, nil
}

// savePRRecords saves PR records to file
func (pt *PRTracker) savePRRecords(records map[string]*PRRecord) error {
	recordsFile := pt.getPRRecordsFile()

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(recordsFile), 0755); err != nil {
		return fmt.Errorf("failed to create records directory: %w", err)
	}

	var lines []string
	for _, record := range records {
		line := fmt.Sprintf("%s\t%d\t%s\t%s\t%s",
			record.PersonaName,
			record.PRNumber,
			record.PRUrl,
			record.CreatedAt.Format(time.RFC3339),
			record.SynthesizedHash)
		lines = append(lines, line)
	}

	content := strings.Join(lines, "\n")
	if len(lines) > 0 {
		content += "\n"
	}

	if err := os.WriteFile(recordsFile, []byte(content), 0644); err != nil {
		return fmt.Errorf("failed to write PR records file: %w", err)
	}

	return nil
}

// getPRRecordsFile returns the path to the PR records file
func (pt *PRTracker) getPRRecordsFile() string {
	return filepath.Join(pt.baseDir, "data", "prompt_prs.txt")
}

// CleanupMergedPRs removes tracking records for PRs that have been merged or closed
func (pt *PRTracker) CleanupMergedPRs(ctx context.Context, githubClient interface{}) error {
	records, err := pt.loadPRRecords()
	if err != nil {
		return fmt.Errorf("failed to load PR records: %w", err)
	}

	cleaned := false
	for personaName, record := range records {
		// Check if the PR is still open/pending via GitHub API
		if ghClient, ok := githubClient.(interface {
			GetPRStatus(ctx context.Context, prNumber int) (string, error)
		}); ok {
			status, err := ghClient.GetPRStatus(ctx, record.PRNumber)
			if err != nil {
				pt.logger.Warnf("Failed to check PR #%d status during cleanup: %v", record.PRNumber, err)
				continue
			}

			// If PR is closed or merged, remove from tracking
			if status == "closed" || status == "merged" {
				pt.logger.Infof("Cleanup: PR #%d for %s is %s, removing from tracking", record.PRNumber, personaName, status)
				delete(records, personaName)
				cleaned = true
			}
		}
	}

	if cleaned {
		return pt.savePRRecords(records)
	}

	return nil
}
