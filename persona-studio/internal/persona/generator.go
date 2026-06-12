package persona

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/google/go-github/v57/github"
	"github.com/sirupsen/logrus"

	"github.com/twin2ai/studio/internal/claude"
	"github.com/twin2ai/studio/pkg/models"
)

type Generator struct {
	claude *claude.Client
	logger *logrus.Logger
}

func NewGenerator(claudeClient *claude.Client, logger *logrus.Logger) *Generator {
	return &Generator{
		claude: claudeClient,
		logger: logger,
	}
}

func (g *Generator) ProcessIssue(ctx context.Context, issue *github.Issue) (*models.Persona, error) {
	g.logger.Infof("Processing issue #%d: %s", *issue.Number, *issue.Title)

	// Combine issue title and body for context
	issueContent := fmt.Sprintf("Title: %s\n\nDescription:\n%s",
		*issue.Title,
		getStringValue(issue.Body))

	// Load template
	template, err := g.loadTemplate()
	if err != nil {
		g.logger.Warnf("Failed to load template: %v", err)
		template = ""
	}

	// Generate persona using Claude
	personaContent, err := g.claude.GeneratePersona(ctx, issueContent, template)
	if err != nil {
		return nil, fmt.Errorf("failed to generate persona: %w", err)
	}

	// Extract persona name
	personaName := g.extractPersonaName(personaContent)

	return &models.Persona{
		Name:        personaName,
		Content:     personaContent,
		IssueNumber: *issue.Number,
	}, nil
}

func (g *Generator) extractPersonaName(content string) string {
	patterns := []string{
		`#\s*(?:Persona:|Name:)\s*(.+)`,
		`##\s*(.+)\s*(?:Persona|Profile)`,
		`Name:\s*(.+)`,
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindStringSubmatch(content)
		if len(matches) > 1 {
			return strings.TrimSpace(matches[1])
		}
	}

	return "Unknown Persona"
}

func (g *Generator) loadTemplate() (string, error) {
	data, err := os.ReadFile("templates/persona_template.md")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (g *Generator) RegeneratePersonaWithFeedback(ctx context.Context, issue *github.Issue, existingPersona string, feedback []string) (*models.Persona, error) {
	g.logger.Infof("Regenerating persona for issue #%d with feedback", *issue.Number)

	// Combine issue title and body for context
	issueContent := fmt.Sprintf("Title: %s\n\nDescription:\n%s",
		*issue.Title,
		getStringValue(issue.Body))

	// Load template
	template, err := g.loadTemplate()
	if err != nil {
		g.logger.Warnf("Failed to load template: %v", err)
		template = ""
	}

	// Create enhanced prompt with feedback
	feedbackSection := g.formatFeedback(feedback)
	enhancedPrompt := fmt.Sprintf(`%s

IMPORTANT: Please address the following feedback and regenerate the persona:

%s

Previous persona version:
%s

Please create an improved version that addresses all the feedback points above.`,
		issueContent, feedbackSection, existingPersona)

	// Generate persona using Claude with feedback
	personaContent, err := g.claude.GeneratePersona(ctx, enhancedPrompt, template)
	if err != nil {
		return nil, fmt.Errorf("failed to regenerate persona: %w", err)
	}

	// Extract persona name
	personaName := g.extractPersonaName(personaContent)

	return &models.Persona{
		Name:        personaName,
		Content:     personaContent,
		IssueNumber: *issue.Number,
	}, nil
}

func (g *Generator) formatFeedback(feedback []string) string {
	if len(feedback) == 0 {
		return "No specific feedback provided."
	}

	var formatted strings.Builder
	for i, comment := range feedback {
		formatted.WriteString(fmt.Sprintf("%d. %s\n", i+1, comment))
	}
	return formatted.String()
}

func (g *Generator) AnalyzeComments(comments []*github.IssueComment) []string {
	var feedback []string

	for _, comment := range comments {
		if comment.Body == nil {
			continue
		}

		body := *comment.Body

		// Skip Studio's own comments
		if strings.Contains(body, "Studio") {
			continue
		}

		// Look for feedback keywords that indicate regeneration is needed
		if g.ContainsFeedbackKeywords(body) {
			feedback = append(feedback, body)
		}
	}

	return feedback
}

func (g *Generator) ContainsFeedbackKeywords(comment string) bool {
	lowerComment := strings.ToLower(comment)

	feedbackKeywords := []string{
		"truncated",
		"incomplete",
		"missing",
		"too short",
		"needs more",
		"expand",
		"add more",
		"please include",
		"could you add",
		"lacking",
		"insufficient",
		"update",
		"change",
		"improve",
		"regenerate",
		"redo",
		"revise",
	}

	for _, keyword := range feedbackKeywords {
		if strings.Contains(lowerComment, keyword) {
			return true
		}
	}

	return false
}

func getStringValue(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
