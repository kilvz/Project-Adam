package github

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/go-github/v57/github"
)

// PromptResult represents a prompt generation result
type PromptResult struct {
	PromptType  string
	Content     string
	GeneratedAt time.Time
	PersonaName string
	Error       error
}

// PromptPRData represents the data needed to create a prompt update PR
type PromptPRData struct {
	PersonaName   string
	PromptResults []PromptResult
	UpdatedReadme string
	UpdatedStatus string
	IssueNumber   *int // Optional - if this is related to an issue
	IsUpdate      bool // true if updating existing persona, false for new
}

// CreatePromptUpdatePR creates a pull request with generated prompts and updated files
func (c *Client) CreatePromptUpdatePR(ctx context.Context, data PromptPRData) (*github.PullRequest, error) {
	c.logger.Infof("Creating prompt update PR for persona: %s", data.PersonaName)

	// Create branch name
	sanitizedName := strings.ToLower(strings.ReplaceAll(data.PersonaName, " ", "-"))
	sanitizedName = strings.ReplaceAll(sanitizedName, "/", "-")
	timestamp := time.Now().Format("20060102-150405")
	branchName := fmt.Sprintf("prompts/%s-%s", sanitizedName, timestamp)

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

	// Prepare persona folder structure
	folderName := strings.ToLower(strings.ReplaceAll(data.PersonaName, " ", "_"))
	folderName = strings.ReplaceAll(folderName, "/", "_")
	baseFolder := fmt.Sprintf("personas/%s", folderName)

	// Track all file operations
	var fileOperations []struct {
		path    string
		content string
		message string
	}

	// Add prompt files
	for _, result := range data.PromptResults {
		if result.Error != nil {
			c.logger.Warnf("Skipping failed prompt result: %s - %v", result.PromptType, result.Error)
			continue
		}

		filename := c.getPromptFilename(result.PromptType)
		// Remove "prompts/" prefix since we're adding it to the base folder
		filename = strings.TrimPrefix(filename, "prompts/")

		promptPath := fmt.Sprintf("%s/prompts/%s", baseFolder, filename)
		promptContent := c.formatPromptForPR(result)

		fileOperations = append(fileOperations, struct {
			path    string
			content string
			message string
		}{promptPath, promptContent, fmt.Sprintf("%s prompt", c.getPromptDisplayName(result.PromptType))})
	}

	// Add updated README if provided
	if data.UpdatedReadme != "" {
		readmePath := fmt.Sprintf("%s/README.md", baseFolder)
		fileOperations = append(fileOperations, struct {
			path    string
			content string
			message string
		}{readmePath, data.UpdatedReadme, "Updated README with prompt information"})
	}

	// Add updated asset status if provided
	if data.UpdatedStatus != "" {
		statusPath := fmt.Sprintf("%s/.assets_status.json", baseFolder)
		fileOperations = append(fileOperations, struct {
			path    string
			content string
			message string
		}{statusPath, data.UpdatedStatus, "Updated asset generation status"})
	}

	// Create/update all files
	for _, op := range fileOperations {
		fileOpts := &github.RepositoryContentFileOptions{
			Message: github.String(fmt.Sprintf("Update %s for persona: %s", op.message, data.PersonaName)),
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
	prTitle, prBody := c.generatePromptPRContent(data)

	pr := &github.NewPullRequest{
		Title: github.String(prTitle),
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
	labels := []string{"prompts", "automated", "studio"}
	if data.IsUpdate {
		labels = append(labels, "update")
	} else {
		labels = append(labels, "new-prompts")
	}

	_, _, err = c.client.Issues.AddLabelsToIssue(
		ctx, c.personasOwner, c.personasRepo,
		*pullRequest.Number, labels)
	if err != nil {
		c.logger.Warnf("Failed to add labels to PR: %v", err)
	}

	// Comment on original issue if this is related to one
	if data.IssueNumber != nil {
		prURL := pullRequest.GetHTMLURL()
		comment := c.generateIssueComment(data, prURL)

		_, _, err = c.client.Issues.CreateComment(
			ctx, c.issuesOwner, c.issuesRepo, *data.IssueNumber,
			&github.IssueComment{Body: github.String(comment)})
		if err != nil {
			c.logger.Warnf("Failed to comment on issue: %v", err)
		}
	}

	c.logger.Infof("Successfully created prompt update PR: %s", pullRequest.GetHTMLURL())
	return pullRequest, nil
}

// formatPromptForPR formats a prompt result for inclusion in the PR
func (c *Client) formatPromptForPR(result PromptResult) string {
	header := fmt.Sprintf(`# %s

> **Generated:** %s  
> **Persona:** %s  
> **Type:** %s  
> **Source:** synthesized.md  
> **Model:** Gemini 2.0 Flash

---

`, c.getPromptDisplayName(result.PromptType),
		result.GeneratedAt.Format(time.RFC3339),
		result.PersonaName,
		result.PromptType)

	footer := fmt.Sprintf(`

---

*Generated automatically by [Studio](https://github.com/twin2ai/studio) using Gemini 2.0 Flash*  
*Last updated: %s*
`, result.GeneratedAt.Format("2006-01-02 15:04:05 UTC"))

	return header + result.Content + footer
}

// generatePromptPRContent generates the PR title and body
func (c *Client) generatePromptPRContent(data PromptPRData) (string, string) {
	// Generate title
	var title string
	if data.IsUpdate {
		title = fmt.Sprintf("Update prompts for persona: %s", data.PersonaName)
	} else {
		title = fmt.Sprintf("Add generated prompts for persona: %s", data.PersonaName)
	}

	// Count successful prompts
	successfulPrompts := 0
	var platformPrompts []string
	var variationPrompts []string

	for _, result := range data.PromptResults {
		if result.Error == nil {
			successfulPrompts++
			displayName := c.getPromptDisplayName(result.PromptType)

			if c.isPlatformPrompt(result.PromptType) {
				platformPrompts = append(platformPrompts, displayName)
			} else if c.isVariationPrompt(result.PromptType) {
				variationPrompts = append(variationPrompts, displayName)
			}
		}
	}

	// Generate body
	var body strings.Builder

	if data.IsUpdate {
		body.WriteString(fmt.Sprintf("This PR updates the generated prompts for **%s** with the latest content from the synthesized persona.\n\n", data.PersonaName))
	} else {
		body.WriteString(fmt.Sprintf("This PR adds comprehensive generated prompts for **%s** based on the synthesized persona content.\n\n", data.PersonaName))
	}

	body.WriteString("## 🎯 Generated Prompts\n\n")
	body.WriteString(fmt.Sprintf("Successfully generated **%d prompts** optimized for different platforms and use cases:\n\n", successfulPrompts))

	if len(platformPrompts) > 0 {
		body.WriteString("### Platform-Specific Prompts\n")
		for _, prompt := range platformPrompts {
			body.WriteString(fmt.Sprintf("- ✅ **%s**\n", prompt))
		}
		body.WriteString("\n")
	}

	if len(variationPrompts) > 0 {
		body.WriteString("### Prompt Variations\n")
		for _, prompt := range variationPrompts {
			body.WriteString(fmt.Sprintf("- ✅ **%s**\n", prompt))
		}
		body.WriteString("\n")
	}

	body.WriteString("## 📁 Files Updated\n\n")
	body.WriteString("- `prompts/` - All generated prompt files\n")
	if data.UpdatedReadme != "" {
		body.WriteString("- `README.md` - Updated with prompt links and usage instructions\n")
	}
	if data.UpdatedStatus != "" {
		body.WriteString("- `.assets_status.json` - Updated asset generation tracking\n")
	}
	body.WriteString("\n")

	body.WriteString("## 🚀 How to Use\n\n")
	body.WriteString("1. **Browse the prompts** in the `prompts/` directory\n")
	body.WriteString("2. **Copy the appropriate prompt** for your AI platform\n")
	body.WriteString("3. **Paste into your AI interface** (ChatGPT, Claude, etc.)\n")
	body.WriteString("4. **Start conversing** with the persona\n\n")

	body.WriteString("## ⚙️ Generation Details\n\n")
	body.WriteString(fmt.Sprintf("- **Model:** Gemini 2.0 Flash\n"))
	body.WriteString(fmt.Sprintf("- **Generated:** %s\n", time.Now().Format("2006-01-02 15:04:05 UTC")))
	body.WriteString(fmt.Sprintf("- **Source:** synthesized.md\n"))
	body.WriteString(fmt.Sprintf("- **Templates:** Platform-specific optimization\n\n"))

	if data.IssueNumber != nil {
		body.WriteString(fmt.Sprintf("## 🔗 Related Issue\n\nCloses #%d\n\n", *data.IssueNumber))
	}

	body.WriteString("---\n")
	body.WriteString("*This PR was created automatically by [Studio](https://github.com/twin2ai/studio) - Multi-AI Persona Generation Pipeline*")

	return title, body.String()
}

// generateIssueComment generates a comment for the original issue
func (c *Client) generateIssueComment(data PromptPRData, prURL string) string {
	successfulPrompts := 0
	for _, result := range data.PromptResults {
		if result.Error == nil {
			successfulPrompts++
		}
	}

	var comment strings.Builder
	comment.WriteString("🎯 **Prompt Generation Complete!**\n\n")
	comment.WriteString(fmt.Sprintf("Generated **%d optimized prompts** for **%s** and created a pull request with all the files.\n\n", successfulPrompts, data.PersonaName))

	comment.WriteString("## Generated Prompts Include:\n")
	comment.WriteString("- Platform-specific prompts (ChatGPT, Claude, Gemini, Discord, Character.AI)\n")
	comment.WriteString("- Condensed prompt-ready version for quick use\n")
	comment.WriteString("- Alternative variations and facets\n\n")

	comment.WriteString(fmt.Sprintf("**📋 Review and merge the pull request:** %s\n\n", prURL))

	comment.WriteString("Once merged, all prompts will be available in the persona's `prompts/` directory for immediate use!")

	return comment.String()
}

// Helper methods for prompt type handling

// getPromptFilename returns the output filename for a prompt type
func (c *Client) getPromptFilename(promptType string) string {
	switch promptType {
	case "chatgpt":
		return "prompts/chatgpt.md"
	case "claude":
		return "prompts/claude.md"
	case "gemini":
		return "prompts/gemini.md"
	case "discord":
		return "prompts/discord.md"
	case "characterai":
		return "prompts/characterai.md"
	case "condensed":
		return "prompts/condensed.md"
	case "alternative":
		return "prompts/alternative.md"
	default:
		return fmt.Sprintf("prompts/%s.md", promptType)
	}
}

// getPromptDisplayName returns a human-readable name for a prompt type
func (c *Client) getPromptDisplayName(promptType string) string {
	switch promptType {
	case "chatgpt":
		return "ChatGPT System Prompt"
	case "claude":
		return "Claude System Prompt"
	case "gemini":
		return "Gemini System Prompt"
	case "discord":
		return "Discord Bot Personality"
	case "characterai":
		return "Character.AI Character"
	case "condensed":
		return "Condensed Prompt-Ready Version"
	case "alternative":
		return "Alternative Variations"
	default:
		return promptType
	}
}

// isPlatformPrompt returns true if the prompt type is platform-specific
func (c *Client) isPlatformPrompt(promptType string) bool {
	switch promptType {
	case "chatgpt", "claude", "gemini", "discord", "characterai":
		return true
	default:
		return false
	}
}

// isVariationPrompt returns true if the prompt type is a variation
func (c *Client) isVariationPrompt(promptType string) bool {
	switch promptType {
	case "condensed", "alternative":
		return true
	default:
		return false
	}
}
