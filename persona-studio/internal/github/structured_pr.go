package github

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/go-github/v57/github"
	"github.com/twin2ai/studio/internal/assets"
)

// PersonaFiles represents all the files that make up a complete persona package
type PersonaFiles struct {
	// Raw AI outputs
	ClaudeRaw string
	GeminiRaw string
	GrokRaw   string
	GPTRaw    string
	UserRaw   string // Optional user-supplied persona

	// Synthesized version
	FullSynthesis string // The complete synthesized persona

	// Asset tracking
	AssetStatus *assets.AssetStatus // Asset generation status
}

// CreateStructuredPersonaPR creates a pull request with the new folder structure
func (c *Client) CreateStructuredPersonaPR(ctx context.Context, issueNumber int, personaName string, files PersonaFiles) (*github.PullRequest, error) {
	// Create branch name
	sanitizedName := strings.ToLower(strings.ReplaceAll(personaName, " ", "-"))
	sanitizedName = strings.ReplaceAll(sanitizedName, "/", "-")
	branchName := fmt.Sprintf("persona/%s-%d", sanitizedName, issueNumber)

	// Get default branch of personas repo
	repo, _, err := c.client.Repositories.Get(ctx, c.personasOwner, c.personasRepo)
	if err != nil {
		return nil, fmt.Errorf("failed to get personas repo: %w", err)
	}
	defaultBranch := repo.GetDefaultBranch()

	// Get base branch ref
	baseRef, _, err := c.client.Git.GetRef(ctx, c.personasOwner, c.personasRepo, "refs/heads/"+defaultBranch)
	if err != nil {
		return nil, fmt.Errorf("failed to get base ref: %w", err)
	}

	// Create new branch
	newRef := &github.Reference{
		Ref: github.String("refs/heads/" + branchName),
		Object: &github.GitObject{
			SHA: baseRef.Object.SHA,
		},
	}

	_, _, err = c.client.Git.CreateRef(ctx, c.personasOwner, c.personasRepo, newRef)
	if err != nil {
		if strings.Contains(err.Error(), "Reference already exists") {
			c.logger.Warnf("Branch %s already exists, continuing...", branchName)
		} else {
			return nil, fmt.Errorf("failed to create branch: %w", err)
		}
	}

	// Create folder structure
	folderName := strings.ToLower(strings.ReplaceAll(personaName, " ", "_"))
	folderName = strings.ReplaceAll(folderName, "/", "_")
	baseFolder := fmt.Sprintf("personas/%s", folderName)

	// Track all file operations for the commit
	fileOperations := []struct {
		path    string
		content string
		message string
	}{
		// Raw AI outputs
		{fmt.Sprintf("%s/raw/claude.md", baseFolder), files.ClaudeRaw, "Claude's raw output"},
		{fmt.Sprintf("%s/raw/gemini.md", baseFolder), files.GeminiRaw, "Gemini's raw output"},
		{fmt.Sprintf("%s/raw/grok.md", baseFolder), files.GrokRaw, "Grok's raw output"},
		{fmt.Sprintf("%s/raw/gpt.md", baseFolder), files.GPTRaw, "GPT-4's raw output"},

		// Main files
		{fmt.Sprintf("%s/synthesized.md", baseFolder), files.FullSynthesis, "Full synthesized persona"},

		// README for the persona folder
		{fmt.Sprintf("%s/README.md", baseFolder), c.generatePersonaReadme(personaName, issueNumber), "Persona overview"},
	}

	// Add user-supplied persona if provided
	if files.UserRaw != "" {
		fileOperations = append(fileOperations, struct {
			path    string
			content string
			message string
		}{fmt.Sprintf("%s/raw/user_supplied.md", baseFolder), files.UserRaw, "User-supplied persona"})
	}

	// Add asset status file if provided
	if files.AssetStatus != nil {
		statusContent, err := c.generateAssetStatusJSON(files.AssetStatus)
		if err != nil {
			c.logger.Warnf("Failed to generate asset status JSON: %v", err)
		} else {
			fileOperations = append(fileOperations, struct {
				path    string
				content string
				message string
			}{fmt.Sprintf("%s/.assets_status.json", baseFolder), statusContent, "Asset generation status"})
		}
	}

	// Create all files
	for _, op := range fileOperations {
		fileOpts := &github.RepositoryContentFileOptions{
			Message: github.String(fmt.Sprintf("Add %s for persona: %s", op.message, personaName)),
			Content: []byte(op.content),
			Branch:  github.String(branchName),
		}

		// Check if file exists
		existingFile, _, _, err := c.client.Repositories.GetContents(
			ctx, c.personasOwner, c.personasRepo, op.path,
			&github.RepositoryContentGetOptions{Ref: branchName})

		if err == nil && existingFile != nil {
			// File exists, update it
			fileOpts.SHA = existingFile.SHA
			_, _, err = c.client.Repositories.UpdateFile(
				ctx, c.personasOwner, c.personasRepo, op.path, fileOpts)
			if err != nil {
				c.logger.Warnf("Failed to update file %s: %v", op.path, err)
				continue
			}
		} else {
			// File doesn't exist, create it
			_, _, err = c.client.Repositories.CreateFile(
				ctx, c.personasOwner, c.personasRepo, op.path, fileOpts)
			if err != nil {
				c.logger.Warnf("Failed to create file %s: %v", op.path, err)
				continue
			}
		}
		c.logger.Debugf("Created/updated file: %s", op.path)
	}

	// Create pull request
	includesUserPersona := ""
	if files.UserRaw != "" {
		includesUserPersona = "\n- **User-supplied persona** included in synthesis"
	}

	// Build source reference based on whether this is batch processing or issue-based
	sourceRef := ""
	if issueNumber > 0 {
		sourceRef = fmt.Sprintf("Created from issue: %s/%s#%d", c.issuesOwner, c.issuesRepo, issueNumber)
	} else {
		sourceRef = "Created via batch processing"
	}

	prBody := fmt.Sprintf(`This PR adds a comprehensive persona package for: **%s**

## 📁 Structure
This persona includes:
- **Raw outputs** from all 4 AI providers (Claude, Gemini, Grok, GPT-4)%s
- **Synthesized version** combining the best of all outputs

## 📍 Files
- %s/raw/ - Individual AI provider outputs
- %s/synthesized.md - Full synthesized persona
- %s/README.md - Documentation and overview

## 🔗 Source
%s

---
*This is an automated PR created by [Studio](https://github.com/twin2ai/studio)*`,
		personaName, includesUserPersona, baseFolder, baseFolder, baseFolder, sourceRef)

	pr := &github.NewPullRequest{
		Title: github.String(fmt.Sprintf("Add persona package: %s", personaName)),
		Body:  github.String(prBody),
		Head:  github.String(branchName),
		Base:  github.String(defaultBranch),
	}

	pullRequest, _, err := c.client.PullRequests.Create(
		ctx, c.personasOwner, c.personasRepo, pr)
	if err != nil {
		return nil, fmt.Errorf("failed to create PR: %w", err)
	}

	// Add labels to PR
	_, _, err = c.client.Issues.AddLabelsToIssue(
		ctx, c.personasOwner, c.personasRepo,
		*pullRequest.Number, []string{"persona", "automated", "studio", "structured"})
	if err != nil {
		c.logger.Warnf("Failed to add labels to PR: %v", err)
	}

	// Comment on original issue with link to PR
	prURL := pullRequest.GetHTMLURL()
	comment := fmt.Sprintf(`✅ Persona package generated successfully!

📦 **Complete persona package created with:**
- Raw outputs from all 4 AI providers (Claude, Gemini, Grok, GPT-4)
- Synthesized full persona combining the best elements
- Documentation and metadata

View the generated persona package: %s

The persona has been created in the [twin2ai/personas](https://github.com/twin2ai/personas) repository.`, prURL)

	// Only comment on real issues (not batch processing which uses issue number 0)
	if issueNumber > 0 {
		_, _, err = c.client.Issues.CreateComment(
			ctx, c.issuesOwner, c.issuesRepo, issueNumber,
			&github.IssueComment{Body: github.String(comment)})
		if err != nil {
			c.logger.Warnf("Failed to comment on issue: %v", err)
		}
	}

	return pullRequest, nil
}

// generatePersonaReadme creates a README file for the persona folder
func (c *Client) generatePersonaReadme(personaName string, issueNumber int) string {
	return fmt.Sprintf(`# %s Persona

This folder contains a comprehensive persona package generated by Studio.

## Contents

### 📝 Raw Outputs (/raw)
- **claude.md** - Claude Opus 4's interpretation
- **gemini.md** - Gemini 2.5 Pro's interpretation
- **grok.md** - Grok 2's interpretation
- **gpt.md** - GPT-4 Turbo's interpretation
- **user_supplied.md** - User-provided persona (if supplied)

### 🎯 Main Files
- **synthesized.md** - Full synthesized persona combining best elements from all AI providers

## Asset Generation

### Available Asset Types
The following assets can be automatically generated based on the synthesized persona:

- **Prompt-Ready Version** - Condensed version optimized for AI prompts
- **Platform Adaptations** - Platform-specific versions (Discord, Telegram, etc.)
- **Voice Clone Config** - Voice synthesis parameters
- **Image Avatar Config** - Avatar generation settings
- **Chatbot Config** - Deployment configuration
- **API Endpoint Config** - API integration settings

### Triggering Asset Generation
To generate additional assets, add trigger markers to this README:

%s

### Asset Status
Check **.assets_status.json** for current generation status and timestamps.

## Usage

1. **For immediate use**: Use synthesized.md for the complete persona
2. **For analysis**: Compare /raw outputs from different AI providers
3. **For assets**: Add generation triggers above to create additional formats

## Source
Generated from issue #%d in the twin2ai/personas repository.

---
*Created by [Studio](https://github.com/twin2ai/studio) - Multi-AI Persona Generation Pipeline*`,
		personaName, generateAssetTriggerExamples(), issueNumber)
}

// generateAssetTriggerExamples creates example trigger markers for the README
func generateAssetTriggerExamples() string {
	return `
<!-- To generate all platform and variation prompts, add this marker: -->
<!-- GENERATE:prompts -->

<!-- To generate only platform-specific prompts (ChatGPT, Claude, etc.), add this marker: -->
<!-- GENERATE:platform_prompts -->

<!-- To generate only variation prompts (condensed, alternative), add this marker: -->
<!-- GENERATE:variation_prompts -->

<!-- To generate other assets, add these markers: -->
<!-- GENERATE:prompt_ready -->
<!-- GENERATE:platform_adaptations -->
<!-- GENERATE:voice_clone -->
<!-- GENERATE:image_avatar -->
<!-- GENERATE:chatbot_config -->
<!-- GENERATE:api_endpoint -->`
}

// generateAssetStatusJSON creates JSON content for the asset status file
func (c *Client) generateAssetStatusJSON(status *assets.AssetStatus) (string, error) {
	// Ensure timestamp is set
	if status.LastSynthesizedUpdate.IsZero() {
		status.LastSynthesizedUpdate = time.Now()
	}

	// Initialize maps if nil
	if status.AssetGenerationFlags == nil {
		status.AssetGenerationFlags = make(map[string]bool)
	}
	if status.Metadata == nil {
		status.Metadata = make(map[string]string)
	}

	// Add creation metadata
	status.Metadata["created_by"] = "studio"
	status.Metadata["created_at"] = time.Now().Format(time.RFC3339)

	// Convert to JSON with proper formatting
	return assets.SerializeAssetStatus(status)
}
