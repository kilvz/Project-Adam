package github

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/go-github/v57/github"
)

// UpdatePersonaWithUserInput creates a PR to update an existing persona with user-provided content
func (c *Client) UpdatePersonaWithUserInput(ctx context.Context, personaName string, existingPersona string, userPersona string, synthesizedPersona string) (*github.PullRequest, error) {
	// Create branch name for update
	sanitizedName := strings.ToLower(strings.ReplaceAll(personaName, " ", "-"))
	sanitizedName = strings.ReplaceAll(sanitizedName, "/", "-")
	timestamp := fmt.Sprintf("%d", time.Now().Unix())
	branchName := fmt.Sprintf("update-persona/%s-%s", sanitizedName, timestamp)

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
		return nil, fmt.Errorf("failed to create branch: %w", err)
	}

	// Create file path
	fileName := strings.ToLower(strings.ReplaceAll(personaName, " ", "_"))
	fileName = strings.ReplaceAll(fileName, "/", "_")

	// For structured personas, update the synthesized.md file
	filePath := fmt.Sprintf("personas/%s/synthesized.md", fileName)

	// Check if this is a structured persona folder
	_, _, _, err = c.client.Repositories.GetContents(
		ctx, c.personasOwner, c.personasRepo, filePath,
		&github.RepositoryContentGetOptions{Ref: defaultBranch})

	if err != nil {
		// Fallback to flat structure
		filePath = fmt.Sprintf("personas/%s.md", fileName)
	}

	// Update the file
	fileOpts := &github.RepositoryContentFileOptions{
		Message: github.String(fmt.Sprintf("Update persona: %s with user input", personaName)),
		Content: []byte(synthesizedPersona),
		Branch:  github.String(branchName),
	}

	// Get existing file SHA
	existingFile, _, _, err := c.client.Repositories.GetContents(
		ctx, c.personasOwner, c.personasRepo, filePath,
		&github.RepositoryContentGetOptions{Ref: defaultBranch})

	if err != nil {
		return nil, fmt.Errorf("failed to get existing persona file: %w", err)
	}

	if existingFile != nil {
		fileOpts.SHA = existingFile.SHA
	}

	// Update the file
	_, _, err = c.client.Repositories.UpdateFile(
		ctx, c.personasOwner, c.personasRepo, filePath, fileOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to update file: %w", err)
	}

	// Create pull request
	prBody := fmt.Sprintf(`This PR updates the persona: **%s**

## 📝 Update Type
User-provided persona synthesized with existing version

## 🔄 Changes
- Synthesized user-provided content with existing persona
- Updated using Gemini synthesis with temperature 0.3

## 💡 Process
1. Existing persona retrieved from repository
2. User-provided persona submitted for synthesis
3. Gemini combined both versions into improved synthesis
4. Only the synthesized version is updated (no raw files or adaptations)

---
*This is an automated PR created by [Studio](https://github.com/twin2ai/studio)*`, personaName)

	pr := &github.NewPullRequest{
		Title: github.String(fmt.Sprintf("Update persona: %s", personaName)),
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
		*pullRequest.Number, []string{"persona", "update", "user-input", "studio"})
	if err != nil {
		c.logger.Warnf("Failed to add labels to PR: %v", err)
	}

	return pullRequest, nil
}

// GetExistingPersona retrieves an existing persona from the repository
func (c *Client) GetExistingPersona(ctx context.Context, personaName string) (string, error) {
	fileName := strings.ToLower(strings.ReplaceAll(personaName, " ", "_"))
	fileName = strings.ReplaceAll(fileName, "/", "_")

	// Try structured format first
	filePath := fmt.Sprintf("personas/%s/synthesized.md", fileName)

	fileContent, _, _, err := c.client.Repositories.GetContents(
		ctx, c.personasOwner, c.personasRepo, filePath,
		&github.RepositoryContentGetOptions{})

	if err != nil {
		// Try flat format
		filePath = fmt.Sprintf("personas/%s.md", fileName)
		fileContent, _, _, err = c.client.Repositories.GetContents(
			ctx, c.personasOwner, c.personasRepo, filePath,
			&github.RepositoryContentGetOptions{})

		if err != nil {
			return "", fmt.Errorf("persona not found: %w", err)
		}
	}

	if fileContent == nil {
		return "", fmt.Errorf("persona file is empty")
	}

	content, err := fileContent.GetContent()
	if err != nil {
		return "", fmt.Errorf("failed to decode file content: %w", err)
	}

	return content, nil
}
