package pipeline

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/go-github/v57/github"
	githubclient "github.com/twin2ai/studio/internal/github"
	"github.com/twin2ai/studio/internal/parser"
	"github.com/twin2ai/studio/pkg/models"
)

// processNewIssuesWithStructure processes new issues and creates structured PRs
func (p *Pipeline) processNewIssuesWithStructure(ctx context.Context) error {
	// Get issues tagged for persona creation
	issues, err := p.github.GetPersonaIssues(ctx)
	if err != nil {
		return fmt.Errorf("failed to get issues: %w", err)
	}

	for _, issue := range issues {
		if issue.Number == nil {
			continue
		}

		// Skip if already processed
		if p.processed[*issue.Number] {
			p.logger.Infof("Issue #%d already processed, skipping", *issue.Number)
			continue
		}

		// Parse the issue using the new template format
		parsedIssue, parseErr := parser.ParsePersonaIssue(issue)
		if parseErr != nil {
			p.logger.Errorf("Failed to parse issue #%d: %v", *issue.Number, parseErr)

			// Only comment on parsing errors that are about title format
			// Body is now optional, so we don't need to comment about missing content
			if strings.Contains(parseErr.Error(), "title") {
				errorComment := parser.GetParsingErrorComment(parseErr)
				_, _, err := p.github.GetClient().Issues.CreateComment(
					ctx, p.config.GitHub.Owner, p.config.GitHub.Repo, *issue.Number,
					&github.IssueComment{Body: github.String(errorComment)})
				if err != nil {
					p.logger.Warnf("Failed to comment parsing error on issue #%d: %v", *issue.Number, err)
				}
			}

			// Mark as processed to avoid repeated error comments
			p.processed[*issue.Number] = true
			if err := p.saveProcessedIssue(*issue.Number); err != nil {
				p.logger.Warnf("Failed to save processed issue: %v", err)
			}
			continue
		}

		// Update the issue with parsed content for generation
		enhancedIssue := &github.Issue{
			Number: issue.Number,
			Title:  github.String(parsedIssue.FullName),
			Body:   github.String(parsedIssue.FormatForPrompt()),
		}

		// Generate persona package using multi-provider approach (with user persona if provided)
		persona, files, err := p.multiGenerator.ProcessIssueWithStructureAndUser(ctx, enhancedIssue, parsedIssue.UserPersona)
		if err != nil {
			p.logger.Errorf("Failed to generate persona package for issue #%d: %v",
				*issue.Number, err)
			continue
		}

		// Create structured PR
		pr, err := p.github.CreateStructuredPersonaPR(ctx,
			persona.IssueNumber,
			persona.Name,
			*files)
		if err != nil {
			p.logger.Errorf("Failed to create structured PR for issue #%d: %v",
				*issue.Number, err)

			// Check if PR already exists (common error)
			if strings.Contains(err.Error(), "A pull request already exists") {
				p.logger.Infof("PR already exists for issue #%d, marking as processed", *issue.Number)
				// Mark as processed to avoid repeated attempts
				p.processed[*issue.Number] = true
				if saveErr := p.saveProcessedIssue(*issue.Number); saveErr != nil {
					p.logger.Warnf("Failed to save processed issue: %v", saveErr)
				}
			}
			continue
		}

		p.logger.Infof("Created structured PR #%d for issue #%d",
			*pr.Number, *issue.Number)

		// Mark as processed
		p.processed[*issue.Number] = true
		if err := p.saveProcessedIssue(*issue.Number); err != nil {
			p.logger.Warnf("Failed to save processed issue: %v", err)
		}
	}

	return nil
}

// processPRCommentsWithStructure processes PR comments and updates structured personas
func (p *Pipeline) processPRCommentsWithStructure(ctx context.Context) error {
	// Get all open Studio PRs
	prs, err := p.github.GetPersonaPullRequests(ctx)
	if err != nil {
		return fmt.Errorf("failed to get PRs: %w", err)
	}

	if len(prs) == 0 {
		p.logger.Info("No open Studio PRs found - skipping comment processing")
		return nil
	}

	for _, pr := range prs {
		if pr.Number == nil {
			continue
		}

		// Verify PR is open
		if pr.State != nil && *pr.State != "open" {
			p.logger.Infof("Skipping closed PR #%d", *pr.Number)
			continue
		}

		p.logger.Infof("Processing open PR #%d for comments", *pr.Number)

		// Get comments for this PR
		comments, err := p.github.GetPRComments(ctx, *pr.Number)
		if err != nil {
			p.logger.Errorf("Failed to get comments for PR #%d: %v", *pr.Number, err)
			continue
		}

		// Analyze comments for feedback
		feedback := p.generator.AnalyzeComments(comments)
		p.logger.Infof("PR #%d has %d feedback comments", *pr.Number, len(feedback))
		if len(feedback) == 0 {
			continue
		}

		// Check if we've already processed these comments
		unprocessedFeedback := p.filterUnprocessedComments(feedback, *pr.Number, comments)
		p.logger.Infof("PR #%d has %d unprocessed feedback comments", *pr.Number, len(unprocessedFeedback))
		if len(unprocessedFeedback) == 0 {
			continue
		}

		// Mark comments as being processed immediately
		for _, comment := range comments {
			if comment.Body != nil && comment.ID != nil && p.generator.ContainsFeedbackKeywords(*comment.Body) {
				commentKey := fmt.Sprintf("%d-%d", *pr.Number, *comment.ID)
				if !p.processedComments[commentKey] {
					p.markCommentAsProcessed(*pr.Number, *comment.ID)
					if err := p.saveProcessedComment(*pr.Number, *comment.ID); err != nil {
						p.logger.Warnf("Failed to save processed comment: %v", err)
					}
				}
			}
		}

		// Find the original issue that created this PR
		originalIssue, err := p.findOriginalIssue(ctx, pr)
		if err != nil {
			p.logger.Errorf("Failed to find original issue for PR #%d: %v", *pr.Number, err)
			continue
		}

		// Get existing persona content from the synthesized file
		existingPersona, err := p.getExistingStructuredPersonaContent(ctx, pr)
		if err != nil {
			p.logger.Errorf("Failed to get existing persona for PR #%d: %v", *pr.Number, err)
			continue
		}

		// Regenerate persona package with feedback
		updatedPersona, updatedFiles, err := p.multiGenerator.RegeneratePersonaWithStructuredFeedback(ctx, originalIssue, existingPersona, unprocessedFeedback)
		if err != nil {
			p.logger.Errorf("Failed to regenerate persona package for PR #%d: %v", *pr.Number, err)
			continue
		}

		// Update all files in the PR
		err = p.updateStructuredPR(ctx, pr, updatedPersona, updatedFiles)
		if err != nil {
			p.logger.Errorf("Failed to update structured PR #%d: %v", *pr.Number, err)
			continue
		}

		p.logger.Infof("Updated structured PR #%d with regenerated persona package based on feedback", *pr.Number)
	}

	return nil
}

// getExistingStructuredPersonaContent retrieves the synthesized persona from a structured PR
func (p *Pipeline) getExistingStructuredPersonaContent(ctx context.Context, pr *github.PullRequest) (string, error) {
	if pr.Head == nil || pr.Head.Ref == nil {
		return "", fmt.Errorf("PR head branch information is missing")
	}

	branchName := *pr.Head.Ref

	// Extract persona name from branch name (format: persona/name-123)
	parts := strings.Split(branchName, "/")
	if len(parts) < 2 {
		return "", fmt.Errorf("invalid branch name format: %s", branchName)
	}

	nameWithIssue := parts[1]
	lastDash := strings.LastIndex(nameWithIssue, "-")
	if lastDash == -1 {
		return "", fmt.Errorf("invalid persona branch name: %s", nameWithIssue)
	}

	personaName := strings.ReplaceAll(nameWithIssue[:lastDash], "-", "_")
	synthesizedPath := fmt.Sprintf("personas/%s/synthesized.md", personaName)

	// Get the synthesized file content
	fileContent, _, _, err := p.github.GetClient().Repositories.GetContents(
		ctx,
		p.config.GitHub.PersonasOwner,
		p.config.GitHub.PersonasRepo,
		synthesizedPath,
		&github.RepositoryContentGetOptions{Ref: branchName})

	if err != nil {
		return "", fmt.Errorf("failed to get synthesized persona content: %w", err)
	}

	if fileContent == nil {
		return "", fmt.Errorf("synthesized persona file not found at %s", synthesizedPath)
	}

	content, err := fileContent.GetContent()
	if err != nil {
		return "", fmt.Errorf("failed to decode file content: %w", err)
	}

	return content, nil
}

// updateStructuredPR updates all files in a structured PR
func (p *Pipeline) updateStructuredPR(ctx context.Context, pr *github.PullRequest, persona *models.Persona, files *githubclient.PersonaFiles) error {
	if pr.Head == nil || pr.Head.Ref == nil {
		return fmt.Errorf("PR head branch information is missing")
	}

	branchName := *pr.Head.Ref

	// Extract persona folder name
	parts := strings.Split(branchName, "/")
	if len(parts) < 2 {
		return fmt.Errorf("invalid branch name format: %s", branchName)
	}

	nameWithIssue := parts[1]
	lastDash := strings.LastIndex(nameWithIssue, "-")
	if lastDash == -1 {
		return fmt.Errorf("invalid persona branch name: %s", nameWithIssue)
	}

	folderName := strings.ReplaceAll(nameWithIssue[:lastDash], "-", "_")
	baseFolder := fmt.Sprintf("personas/%s", folderName)

	// Define all files to update
	fileUpdates := []struct {
		path    string
		content string
		message string
	}{
		// Raw AI outputs
		{fmt.Sprintf("%s/raw/claude.md", baseFolder), files.ClaudeRaw, "Update Claude's raw output"},
		{fmt.Sprintf("%s/raw/gemini.md", baseFolder), files.GeminiRaw, "Update Gemini's raw output"},
		{fmt.Sprintf("%s/raw/grok.md", baseFolder), files.GrokRaw, "Update Grok's raw output"},
		{fmt.Sprintf("%s/raw/gpt.md", baseFolder), files.GPTRaw, "Update GPT-4's raw output"},

		// Main files
		{fmt.Sprintf("%s/synthesized.md", baseFolder), files.FullSynthesis, "Update synthesized persona"},
	}

	// Update each file
	for _, update := range fileUpdates {
		fileOpts := &github.RepositoryContentFileOptions{
			Message: github.String(fmt.Sprintf("%s (addressing feedback)", update.message)),
			Content: []byte(update.content),
			Branch:  github.String(branchName),
		}

		// Get existing file to get SHA
		existingFile, _, _, err := p.github.GetClient().Repositories.GetContents(
			ctx,
			p.config.GitHub.PersonasOwner,
			p.config.GitHub.PersonasRepo,
			update.path,
			&github.RepositoryContentGetOptions{Ref: branchName})

		if err == nil && existingFile != nil {
			fileOpts.SHA = existingFile.SHA
			_, _, err = p.github.GetClient().Repositories.UpdateFile(
				ctx,
				p.config.GitHub.PersonasOwner,
				p.config.GitHub.PersonasRepo,
				update.path,
				fileOpts)
			if err != nil {
				p.logger.Warnf("Failed to update file %s: %v", update.path, err)
				continue
			}
		} else {
			p.logger.Warnf("File %s not found, skipping update", update.path)
			continue
		}

		p.logger.Debugf("Updated file: %s", update.path)
	}

	// Add a comment to the PR indicating the update
	comment := fmt.Sprintf(`🔄 **Persona Package Updated**

All files in the persona package have been regenerated to address the feedback provided:
- Raw outputs from all 4 AI providers
- Synthesized persona

The complete package has been updated to incorporate your suggestions.

---
*Updated automatically by [Studio](https://github.com/twin2ai/studio)*`)

	_, _, err := p.github.GetClient().Issues.CreateComment(
		ctx,
		p.config.GitHub.PersonasOwner,
		p.config.GitHub.PersonasRepo,
		*pr.Number,
		&github.IssueComment{Body: github.String(comment)})
	if err != nil {
		p.logger.Warnf("Failed to comment on PR after update: %v", err)
	}

	return nil
}

// runWithStructure runs the pipeline with structured PR support
func (p *Pipeline) runWithStructure(ctx context.Context) error {
	p.logger.Info("Running structured pipeline iteration")

	// Process new issues first
	if err := p.processNewIssuesWithStructure(ctx); err != nil {
		p.logger.Errorf("Failed to process new issues: %v", err)
	}

	// Process PR comments for existing personas
	if err := p.processPRCommentsWithStructure(ctx); err != nil {
		p.logger.Errorf("Failed to process PR comments: %v", err)
	}

	return nil
}
