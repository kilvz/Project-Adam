package pipeline

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/go-co-op/gocron"
	"github.com/google/go-github/v57/github"
	"github.com/sirupsen/logrus"

	"github.com/twin2ai/studio/internal/claude"
	"github.com/twin2ai/studio/internal/config"
	"github.com/twin2ai/studio/internal/gemini"
	githubclient "github.com/twin2ai/studio/internal/github"
	"github.com/twin2ai/studio/internal/gpt"
	"github.com/twin2ai/studio/internal/grok"
	"github.com/twin2ai/studio/internal/multiprovider"
	"github.com/twin2ai/studio/internal/persona"
)

type Pipeline struct {
	config            *config.Config
	github            *githubclient.Client
	generator         *persona.Generator
	multiGenerator    *multiprovider.Generator
	promptIntegration *PromptPipelineIntegration
	logger            *logrus.Logger
	processed         map[int]bool
	processedComments map[string]bool // Track processed comments by PR#-CommentID
}

func New(cfg *config.Config, logger *logrus.Logger) (*Pipeline, error) {
	// Create GitHub client
	githubClient := githubclient.NewClient(
		cfg.GitHub.Token,
		cfg.GitHub.Owner,
		cfg.GitHub.Repo,
		cfg.GitHub.PersonasOwner,
		cfg.GitHub.PersonasRepo,
		cfg.GitHub.PersonaLabel,
		logger,
	)

	// Create AI clients
	claudeClient := claude.NewClient(cfg.AI.Claude.APIKey, cfg.AI.Claude.Model, logger)
	geminiClient := gemini.NewClient(cfg.AI.Gemini.APIKey, cfg.AI.Gemini.Model, logger)
	grokClient := grok.NewClient(cfg.AI.Grok.APIKey, cfg.AI.Grok.Model, logger)
	gptClient := gpt.NewClient(cfg.AI.GPT.APIKey, cfg.AI.GPT.Model, logger)

	// Create persona generators
	generator := persona.NewGenerator(claudeClient, logger)
	multiGenerator := multiprovider.NewGenerator(claudeClient, geminiClient, grokClient, gptClient, logger)

	// Create prompt integration (enable if Gemini API key is available)
	promptEnabled := cfg.AI.Gemini.APIKey != ""
	promptIntegration := NewPromptPipelineIntegration(geminiClient, githubClient, logger, ".", promptEnabled)

	p := &Pipeline{
		config:            cfg,
		github:            githubClient,
		generator:         generator,
		multiGenerator:    multiGenerator,
		promptIntegration: promptIntegration,
		logger:            logger,
		processed:         make(map[int]bool),
		processedComments: make(map[string]bool),
	}

	// Load processed issues and comments
	if err := p.loadProcessedIssues(); err != nil {
		logger.Warnf("Failed to load processed issues: %v", err)
	}
	if err := p.loadProcessedComments(); err != nil {
		logger.Warnf("Failed to load processed comments: %v", err)
	}

	return p, nil
}

func (p *Pipeline) Start(ctx context.Context) error {
	// Run once immediately
	if err := p.run(ctx); err != nil {
		p.logger.Errorf("Initial run failed: %v", err)
	}

	// Setup scheduler
	s := gocron.NewScheduler(time.UTC)

	_, err := s.Every(p.config.Pipeline.PollInterval).Do(func() {
		if err := p.run(ctx); err != nil {
			p.logger.Errorf("Pipeline run failed: %v", err)
		}
	})

	if err != nil {
		return fmt.Errorf("failed to schedule pipeline: %w", err)
	}

	s.StartAsync()

	// Wait for context cancellation
	<-ctx.Done()
	s.Stop()

	return nil
}

func (p *Pipeline) run(ctx context.Context) error {
	p.logger.Info("Running pipeline iteration")

	// Process update requests first
	if err := p.processUpdateRequests(ctx); err != nil {
		p.logger.Errorf("Failed to process update requests: %v", err)
	}

	// Use structured pipeline by default for new personas
	if err := p.runWithStructure(ctx); err != nil {
		p.logger.Errorf("Structured pipeline run failed: %v", err)
	}

	// Process prompt generation triggers
	if err := p.promptIntegration.ProcessPromptGeneration(ctx); err != nil {
		p.logger.Errorf("Prompt generation processing failed: %v", err)
	}

	return nil
}

func (p *Pipeline) processUpdateRequests(ctx context.Context) error {
	// Get issues tagged for persona updates
	opts := &github.IssueListByRepoOptions{
		State:  "open",
		Labels: []string{"update-persona"},
	}

	issues, _, err := p.github.GetClient().Issues.ListByRepo(ctx, p.config.GitHub.Owner, p.config.GitHub.Repo, opts)
	if err != nil {
		return fmt.Errorf("failed to fetch update issues: %w", err)
	}

	p.logger.Infof("Found %d update requests", len(issues))

	for _, issue := range issues {
		if issue.Number == nil {
			continue
		}

		// Skip if already processed
		if p.processed[*issue.Number] {
			p.logger.Infof("Update issue #%d already processed, skipping", *issue.Number)
			continue
		}

		// Parse update request
		request, err := ParseUpdateRequest(issue)
		if err != nil {
			p.logger.Errorf("Failed to parse update request from issue #%d: %v", *issue.Number, err)

			// Comment on the issue with error
			errorComment := fmt.Sprintf(`❌ **Unable to Process Update Request**

%s

**Required Format:**
- Title: "Update Persona: [Existing Persona Name]"
- Body: Your updated persona content (optionally wrapped in [[[ ]]] markers)

Please ensure the persona name matches an existing persona exactly.

---
*[Studio](https://github.com/twin2ai/studio) - Multi-AI Persona Generation Pipeline*`, err.Error())

			_, _, commentErr := p.github.GetClient().Issues.CreateComment(
				ctx, p.config.GitHub.Owner, p.config.GitHub.Repo, *issue.Number,
				&github.IssueComment{Body: github.String(errorComment)})
			if commentErr != nil {
				p.logger.Warnf("Failed to comment error on issue #%d: %v", *issue.Number, commentErr)
			}

			// Mark as processed to avoid repeated error comments
			p.processed[*issue.Number] = true
			if err := p.saveProcessedIssue(*issue.Number); err != nil {
				p.logger.Warnf("Failed to save processed issue: %v", err)
			}
			continue
		}

		// Process the update
		if err := p.ProcessPersonaUpdate(ctx, *request); err != nil {
			p.logger.Errorf("Failed to process persona update: %v", err)

			// Comment on the issue with error
			errorComment := fmt.Sprintf(`❌ **Failed to Update Persona**

%s

Please check that:
- The persona name matches an existing persona
- The repository is accessible
- Your content is valid

---
*[Studio](https://github.com/twin2ai/studio) - Multi-AI Persona Generation Pipeline*`, err.Error())

			_, _, commentErr := p.github.GetClient().Issues.CreateComment(
				ctx, p.config.GitHub.Owner, p.config.GitHub.Repo, *issue.Number,
				&github.IssueComment{Body: github.String(errorComment)})
			if commentErr != nil {
				p.logger.Warnf("Failed to comment error on issue #%d: %v", *issue.Number, commentErr)
			}
		} else {
			// Success comment
			successComment := fmt.Sprintf(`✅ **Persona Update Submitted**

Successfully synthesized your update with the existing persona for **%s**.

A pull request has been created with the updated version. The synthesis combines your input with the existing persona to create an improved version.

**Note**: Only the synthesized version is updated in update requests.

---
*[Studio](https://github.com/twin2ai/studio) - Multi-AI Persona Generation Pipeline*`, request.PersonaName)

			_, _, commentErr := p.github.GetClient().Issues.CreateComment(
				ctx, p.config.GitHub.Owner, p.config.GitHub.Repo, *issue.Number,
				&github.IssueComment{Body: github.String(successComment)})
			if commentErr != nil {
				p.logger.Warnf("Failed to comment success on issue #%d: %v", *issue.Number, commentErr)
			}
		}

		// Mark as processed
		p.processed[*issue.Number] = true
		if err := p.saveProcessedIssue(*issue.Number); err != nil {
			p.logger.Warnf("Failed to save processed issue: %v", err)
		}
	}

	return nil
}

func (p *Pipeline) processNewIssues(ctx context.Context) error {
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

		// Generate persona using multi-provider approach
		persona, err := p.multiGenerator.ProcessIssue(ctx, issue)
		if err != nil {
			p.logger.Errorf("Failed to generate persona for issue #%d: %v",
				*issue.Number, err)
			continue
		}

		// Create PR
		pr, err := p.github.CreatePersonaPullRequest(ctx,
			persona.IssueNumber,
			persona.Name,
			persona.Content)
		if err != nil {
			p.logger.Errorf("Failed to create PR for issue #%d: %v",
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

		p.logger.Infof("Created PR #%d for issue #%d",
			*pr.Number, *issue.Number)

		// Mark as processed
		p.processed[*issue.Number] = true
		if err := p.saveProcessedIssue(*issue.Number); err != nil {
			p.logger.Warnf("Failed to save processed issue: %v", err)
		}
	}

	return nil
}

func (p *Pipeline) processPRComments(ctx context.Context) error {
	// Get all open Studio PRs (closed PRs are automatically excluded)
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

		// Verify PR is open (should already be filtered by GitHub API)
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

		// Mark comments as being processed immediately to prevent duplicate processing
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

		// Get existing persona content
		existingPersona, err := p.getExistingPersonaContent(ctx, pr)
		if err != nil {
			p.logger.Errorf("Failed to get existing persona for PR #%d: %v", *pr.Number, err)
			continue
		}

		// Regenerate persona with feedback using multi-provider approach
		updatedPersona, err := p.multiGenerator.RegeneratePersonaWithFeedback(ctx, originalIssue, existingPersona, unprocessedFeedback)
		if err != nil {
			p.logger.Errorf("Failed to regenerate persona for PR #%d: %v", *pr.Number, err)
			continue
		}

		// Update the PR with new persona
		branchName := pr.Head.GetRef()
		filePath, err := p.getActualPersonaFilePath(ctx, pr)
		if err != nil {
			p.logger.Errorf("Failed to get actual file path for PR #%d: %v", *pr.Number, err)
			continue
		}

		err = p.github.UpdatePersonaPR(ctx, *pr.Number, updatedPersona.Name, updatedPersona.Content, branchName, filePath)
		if err != nil {
			p.logger.Errorf("Failed to update PR #%d: %v", *pr.Number, err)
			continue
		}

		// Comments are already marked as processed above

		p.logger.Infof("Updated PR #%d with regenerated persona based on feedback", *pr.Number)
	}

	return nil
}

func (p *Pipeline) loadProcessedIssues() error {
	filePath := filepath.Join(p.config.Pipeline.DataDir, "processed_issues.txt")

	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		issueNum, err := strconv.Atoi(line)
		if err != nil {
			p.logger.Warnf("Invalid issue number in processed file: %s", line)
			continue
		}

		p.processed[issueNum] = true
	}

	return nil
}

func (p *Pipeline) saveProcessedIssue(issueNumber int) error {
	filePath := filepath.Join(p.config.Pipeline.DataDir, "processed_issues.txt")

	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.WriteString(fmt.Sprintf("%d\n", issueNumber))
	return err
}

func (p *Pipeline) loadProcessedComments() error {
	filePath := filepath.Join(p.config.Pipeline.DataDir, "processed_comments.txt")

	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		p.processedComments[line] = true
	}

	return nil
}

func (p *Pipeline) saveProcessedComment(prNumber int, commentID int64) error {
	filePath := filepath.Join(p.config.Pipeline.DataDir, "processed_comments.txt")

	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	commentKey := fmt.Sprintf("%d-%d", prNumber, commentID)
	_, err = f.WriteString(fmt.Sprintf("%s\n", commentKey))
	return err
}

func (p *Pipeline) markCommentAsProcessed(prNumber int, commentID int64) {
	commentKey := fmt.Sprintf("%d-%d", prNumber, commentID)
	p.processedComments[commentKey] = true
}

func (p *Pipeline) filterUnprocessedComments(feedback []string, prNumber int, comments []*github.IssueComment) []string {
	var unprocessed []string

	for _, comment := range comments {
		if comment.Body == nil || comment.ID == nil {
			continue
		}

		// Skip Studio's own comments
		if strings.Contains(*comment.Body, "Studio") || strings.Contains(*comment.Body, "Updated automatically") {
			continue
		}

		commentKey := fmt.Sprintf("%d-%d", prNumber, *comment.ID)

		// Only process if not already processed and contains feedback keywords
		if !p.processedComments[commentKey] && p.generator.ContainsFeedbackKeywords(*comment.Body) {
			unprocessed = append(unprocessed, *comment.Body)
			p.logger.Infof("Found unprocessed feedback comment: %s", commentKey)
		} else if p.processedComments[commentKey] {
			p.logger.Infof("Skipping already processed comment: %s", commentKey)
		}
	}

	return unprocessed
}

func (p *Pipeline) findOriginalIssue(ctx context.Context, pr *github.PullRequest) (*github.Issue, error) {
	// Parse issue number from PR body
	if pr.Body == nil {
		return nil, fmt.Errorf("PR body is empty")
	}

	body := *pr.Body
	// Look for pattern like "Created from issue: owner/repo#123"
	re := regexp.MustCompile(`Created from issue: [^#]+#(\d+)`)
	matches := re.FindStringSubmatch(body)
	if len(matches) < 2 {
		return nil, fmt.Errorf("could not find issue number in PR body")
	}

	issueNumber, err := strconv.Atoi(matches[1])
	if err != nil {
		return nil, fmt.Errorf("invalid issue number: %v", err)
	}

	// Get the issue
	issue, _, err := p.github.GetClient().Issues.Get(ctx, p.config.GitHub.Owner, p.config.GitHub.Repo, issueNumber)
	if err != nil {
		return nil, fmt.Errorf("failed to get issue #%d: %w", issueNumber, err)
	}

	return issue, nil
}

func (p *Pipeline) getExistingPersonaContent(ctx context.Context, pr *github.PullRequest) (string, error) {
	if pr.Head == nil || pr.Head.Ref == nil {
		return "", fmt.Errorf("PR head ref is nil")
	}

	branchName := *pr.Head.Ref

	// Get the list of files in the personas directory for this branch
	_, dirContent, _, err := p.github.GetClient().Repositories.GetContents(
		ctx, p.config.GitHub.PersonasOwner, p.config.GitHub.PersonasRepo, "personas",
		&github.RepositoryContentGetOptions{Ref: branchName})
	if err != nil {
		return "", fmt.Errorf("failed to get personas directory: %w", err)
	}

	// Find the .md file in the personas directory
	var personaFile *github.RepositoryContent
	for _, file := range dirContent {
		if file.Name != nil && strings.HasSuffix(*file.Name, ".md") {
			personaFile = file
			break
		}
	}

	if personaFile == nil {
		return "", fmt.Errorf("no .md file found in personas directory")
	}

	// Get the actual file content
	fileContent, _, _, err := p.github.GetClient().Repositories.GetContents(
		ctx, p.config.GitHub.PersonasOwner, p.config.GitHub.PersonasRepo, *personaFile.Path,
		&github.RepositoryContentGetOptions{Ref: branchName})
	if err != nil {
		return "", fmt.Errorf("failed to get file content: %w", err)
	}

	if fileContent.Content == nil {
		return "", fmt.Errorf("file content is empty")
	}

	content, err := fileContent.GetContent()
	if err != nil {
		return "", fmt.Errorf("failed to decode file content: %w", err)
	}

	return content, nil
}

func (p *Pipeline) getPersonaFilePath(personaName string) string {
	fileName := strings.ToLower(strings.ReplaceAll(personaName, " ", "_"))
	fileName = strings.ReplaceAll(fileName, "/", "_")
	return fmt.Sprintf("personas/%s.md", fileName)
}

func (p *Pipeline) getActualPersonaFilePath(ctx context.Context, pr *github.PullRequest) (string, error) {
	if pr.Head == nil || pr.Head.Ref == nil {
		return "", fmt.Errorf("PR head ref is nil")
	}

	branchName := *pr.Head.Ref

	// Get the list of files in the personas directory for this branch
	_, dirContent, _, err := p.github.GetClient().Repositories.GetContents(
		ctx, p.config.GitHub.PersonasOwner, p.config.GitHub.PersonasRepo, "personas",
		&github.RepositoryContentGetOptions{Ref: branchName})
	if err != nil {
		return "", fmt.Errorf("failed to get personas directory: %w", err)
	}

	// Find the .md file in the personas directory
	for _, file := range dirContent {
		if file.Name != nil && strings.HasSuffix(*file.Name, ".md") {
			return *file.Path, nil
		}
	}

	return "", fmt.Errorf("no .md file found in personas directory")
}
