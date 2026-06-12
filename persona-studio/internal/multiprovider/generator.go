package multiprovider

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/google/go-github/v57/github"
	"github.com/sirupsen/logrus"

	"github.com/twin2ai/studio/internal/claude"
	"github.com/twin2ai/studio/internal/gemini"
	"github.com/twin2ai/studio/internal/gpt"
	"github.com/twin2ai/studio/internal/grok"
	"github.com/twin2ai/studio/pkg/models"
)

type Generator struct {
	claude  *claude.Client
	gemini  *gemini.Client
	grok    *grok.Client
	gpt     *gpt.Client
	logger  *logrus.Logger
	baseDir string
}

type ProviderResponse struct {
	Provider string
	Content  string
	Error    error
}

func NewGenerator(claudeClient *claude.Client, geminiClient *gemini.Client, grokClient *grok.Client, gptClient *gpt.Client, logger *logrus.Logger) *Generator {
	return &Generator{
		claude:  claudeClient,
		gemini:  geminiClient,
		grok:    grokClient,
		gpt:     gptClient,
		logger:  logger,
		baseDir: "artifacts",
	}
}

func (g *Generator) ProcessIssue(ctx context.Context, issue *github.Issue) (*models.Persona, error) {
	g.logger.Infof("Processing issue #%d with multi-provider generation: %s", *issue.Number, *issue.Title)

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

	// Generate personas from all providers in parallel
	responses, err := g.generateFromAllProviders(ctx, issueContent, template)
	if err != nil {
		return nil, fmt.Errorf("failed to generate personas from providers: %w", err)
	}

	// Store individual responses
	if err := g.storeArtifacts(*issue.Number, responses); err != nil {
		g.logger.Warnf("Failed to store artifacts: %v", err)
	}

	// Combine all responses into final persona
	finalPersona, err := g.combinePersonas(ctx, responses)
	if err != nil {
		return nil, fmt.Errorf("failed to combine personas: %w", err)
	}

	// Store combined result
	if err := g.storeCombinedPersona(*issue.Number, finalPersona); err != nil {
		g.logger.Warnf("Failed to store combined persona: %v", err)
	}

	// Use persona name from issue title if available, otherwise try to extract
	personaName := "AI-Generated Persona"
	if issue.Title != nil && *issue.Title != "" {
		personaName = *issue.Title
	} else {
		// Fallback to extraction if no title
		personaName = g.extractPersonaName(finalPersona)
	}

	return &models.Persona{
		Name:        personaName,
		Content:     finalPersona,
		IssueNumber: *issue.Number,
	}, nil
}

func (g *Generator) RegeneratePersonaWithFeedback(ctx context.Context, issue *github.Issue, existingPersona string, feedback []string) (*models.Persona, error) {
	g.logger.Infof("Regenerating persona for issue #%d with feedback using all providers", *issue.Number)

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

	// Generate from all providers with feedback
	responses, err := g.generateFromAllProviders(ctx, enhancedPrompt, template)
	if err != nil {
		return nil, fmt.Errorf("failed to regenerate personas from providers: %w", err)
	}

	// Store individual responses with feedback suffix
	if err := g.storeArtifactsWithSuffix(*issue.Number, responses, "feedback"); err != nil {
		g.logger.Warnf("Failed to store feedback artifacts: %v", err)
	}

	// Combine all responses with feedback prompt
	finalPersona, err := g.combinePersonasWithFeedback(ctx, responses, feedback)
	if err != nil {
		return nil, fmt.Errorf("failed to combine feedback personas: %w", err)
	}

	// Store combined result with feedback suffix
	if err := g.storeCombinedPersonaWithSuffix(*issue.Number, finalPersona, "feedback"); err != nil {
		g.logger.Warnf("Failed to store combined feedback persona: %v", err)
	}

	// Use persona name from issue title if available, otherwise try to extract
	personaName := "AI-Generated Persona"
	if issue.Title != nil && *issue.Title != "" {
		personaName = *issue.Title
	} else {
		// Fallback to extraction if no title
		personaName = g.extractPersonaName(finalPersona)
	}

	return &models.Persona{
		Name:        personaName,
		Content:     finalPersona,
		IssueNumber: *issue.Number,
	}, nil
}

func (g *Generator) generateFromAllProviders(ctx context.Context, issueContent, template string) ([]ProviderResponse, error) {
	// Prepare the full prompt for providers that can handle it
	var fullPrompt string
	if template != "" {
		fullPrompt = fmt.Sprintf("%s\n\nUse this template as a guide:\n%s", issueContent, template)
	} else {
		fullPrompt = issueContent
	}

	// Create a shorter prompt for GPT-4 to avoid context length issues
	shortPrompt := fmt.Sprintf(`%s

Create a detailed user persona based on the above information. Include:
1. Name and demographics
2. Background and goals
3. Pain points and challenges
4. Technical proficiency
5. Behavioral patterns
6. Success criteria

Format as a well-structured Markdown document.`, issueContent)

	// Create channels for responses
	responses := make(chan ProviderResponse, 4)
	var wg sync.WaitGroup

	// Generate from Claude - uses its own template handling
	wg.Add(1)
	go func() {
		defer wg.Done()
		content, err := g.claude.GeneratePersona(ctx, issueContent, template)
		responses <- ProviderResponse{
			Provider: "claude",
			Content:  content,
			Error:    err,
		}
	}()

	// Generate from Gemini - can handle full prompt
	wg.Add(1)
	go func() {
		defer wg.Done()
		content, err := g.gemini.GeneratePersona(ctx, fullPrompt)
		responses <- ProviderResponse{
			Provider: "gemini",
			Content:  content,
			Error:    err,
		}
	}()

	// Generate from Grok - can handle full prompt
	wg.Add(1)
	go func() {
		defer wg.Done()
		content, err := g.grok.GeneratePersona(ctx, fullPrompt)
		responses <- ProviderResponse{
			Provider: "grok",
			Content:  content,
			Error:    err,
		}
	}()

	// Generate from GPT - use shorter prompt to avoid context length issues
	wg.Add(1)
	go func() {
		defer wg.Done()
		content, err := g.gpt.GeneratePersona(ctx, shortPrompt)
		responses <- ProviderResponse{
			Provider: "gpt",
			Content:  content,
			Error:    err,
		}
	}()

	// Wait for all to complete
	wg.Wait()
	close(responses)

	// Collect results
	var results []ProviderResponse
	successCount := 0
	for resp := range responses {
		results = append(results, resp)
		if resp.Error == nil {
			successCount++
			g.logger.Infof("Successfully generated persona from %s", resp.Provider)
		} else {
			g.logger.Errorf("Failed to generate persona from %s: %v", resp.Provider, resp.Error)
		}
	}

	if successCount == 0 {
		return nil, fmt.Errorf("all providers failed to generate personas")
	}

	g.logger.Infof("Generated personas from %d/%d providers", successCount, len(results))
	return results, nil
}

func (g *Generator) combinePersonas(ctx context.Context, responses []ProviderResponse) (string, error) {
	return g.combinePersonasWithUser(ctx, responses, "")
}

func (g *Generator) combinePersonasWithUser(ctx context.Context, responses []ProviderResponse, userPersona string) (string, error) {
	g.logger.Info("Combining personas using Gemini")
	if userPersona != "" {
		g.logger.Info("Including user-supplied persona in synthesis")
	}

	// Filter successful responses
	var successfulResponses []ProviderResponse
	for _, resp := range responses {
		if resp.Error == nil {
			successfulResponses = append(successfulResponses, resp)
		}
	}

	if len(successfulResponses) == 0 {
		return "", fmt.Errorf("no successful persona responses to combine")
	}

	if len(successfulResponses) == 1 && userPersona == "" {
		g.logger.Info("Only one successful response and no user persona, using it directly")
		return successfulResponses[0].Content, nil
	}

	// Load combination prompt template
	combinationPrompt, err := g.loadCombinationPrompt()
	if err != nil {
		g.logger.Warnf("Failed to load combination prompt, using fallback: %v", err)
		combinationPrompt = g.getDefaultCombinationPrompt()
	}

	// If user persona exists, update the prompt to include it
	if userPersona != "" {
		combinationPrompt = strings.Replace(combinationPrompt,
			"INPUT PERSONAS TO COMBINE:",
			"INPUT PERSONAS TO COMBINE (including user-supplied version):",
			1)
	}

	finalPrompt := combinationPrompt

	// Build the personas section
	var personaCount int
	var personas []string
	for _, resp := range successfulResponses {
		if resp.Content == "" {
			continue
		}
		personaCount++
		personas = append(personas, fmt.Sprintf("Persona %d: %s\n<<<\n%s\n>>>", personaCount, resp.Provider, resp.Content))
	}

	// Add user persona if provided
	if userPersona != "" {
		personaCount++
		personas = append(personas, fmt.Sprintf("Persona %d: %s\n<<<\n%s\n>>>", personaCount, "User-Supplied", userPersona))
	}

	finalPrompt = strings.ReplaceAll(finalPrompt, "{{PERSONAS}}", strings.Join(personas, "\n\n"))

	// Use Gemini with lower temperature for synthesis
	combinedPersona, err := g.gemini.GenerateSynthesis(ctx, finalPrompt)
	if err != nil {
		g.logger.Warnf("Failed to combine with Gemini, using best individual response: %v", err)
		// Fallback to the longest response as it's likely most complete
		bestResponse := successfulResponses[0]
		for _, resp := range successfulResponses[1:] {
			if len(resp.Content) > len(bestResponse.Content) {
				bestResponse = resp
			}
		}
		return bestResponse.Content, nil
	}

	return combinedPersona, nil
}

func (g *Generator) storeArtifacts(issueNumber int, responses []ProviderResponse) error {
	timestamp := time.Now().Format("20060102-150405")
	for _, resp := range responses {
		if resp.Error != nil {
			continue
		}

		filename := fmt.Sprintf("issue-%d-%s-%s.md", issueNumber, resp.Provider, timestamp)
		filePath := filepath.Join(g.baseDir, resp.Provider, filename)

		if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
			return fmt.Errorf("failed to create directory: %w", err)
		}

		if err := os.WriteFile(filePath, []byte(resp.Content), 0644); err != nil {
			return fmt.Errorf("failed to write %s artifact: %w", resp.Provider, err)
		}

		g.logger.Infof("Stored %s persona artifact: %s", resp.Provider, filePath)
	}
	return nil
}

func (g *Generator) storeArtifactsWithSuffix(issueNumber int, responses []ProviderResponse, suffix string) error {
	timestamp := time.Now().Format("20060102-150405")
	for _, resp := range responses {
		if resp.Error != nil {
			continue
		}

		filename := fmt.Sprintf("issue-%d-%s-%s-%s.md", issueNumber, resp.Provider, suffix, timestamp)
		filePath := filepath.Join(g.baseDir, resp.Provider, filename)

		if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
			return fmt.Errorf("failed to create directory: %w", err)
		}

		if err := os.WriteFile(filePath, []byte(resp.Content), 0644); err != nil {
			return fmt.Errorf("failed to write %s artifact: %w", resp.Provider, err)
		}

		g.logger.Infof("Stored %s %s persona artifact: %s", resp.Provider, suffix, filePath)
	}
	return nil
}

func (g *Generator) storeCombinedPersona(issueNumber int, persona string) error {
	timestamp := time.Now().Format("20060102-150405")
	filename := fmt.Sprintf("issue-%d-combined-%s.md", issueNumber, timestamp)
	filePath := filepath.Join(g.baseDir, "combined", filename)

	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return fmt.Errorf("failed to create combined directory: %w", err)
	}

	if err := os.WriteFile(filePath, []byte(persona), 0644); err != nil {
		return fmt.Errorf("failed to write combined persona: %w", err)
	}

	g.logger.Infof("Stored combined persona: %s", filePath)
	return nil
}

func (g *Generator) storeCombinedPersonaWithSuffix(issueNumber int, persona, suffix string) error {
	timestamp := time.Now().Format("20060102-150405")
	filename := fmt.Sprintf("issue-%d-combined-%s-%s.md", issueNumber, suffix, timestamp)
	filePath := filepath.Join(g.baseDir, "combined", filename)

	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return fmt.Errorf("failed to create combined directory: %w", err)
	}

	if err := os.WriteFile(filePath, []byte(persona), 0644); err != nil {
		return fmt.Errorf("failed to write combined persona: %w", err)
	}

	g.logger.Infof("Stored combined %s persona: %s", suffix, filePath)
	return nil
}

// extractPersonaName attempts to extract a persona name from generated content
// This is only used as a fallback when the issue title is not available
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

	return "AI-Generated Persona"
}

func (g *Generator) loadTemplate() (string, error) {
	data, err := os.ReadFile("templates/persona_template.md")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (g *Generator) loadCombinationPrompt() (string, error) {
	data, err := os.ReadFile("prompts/persona_combination.txt")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (g *Generator) getDefaultCombinationPrompt() string {
	return `You are tasked with creating the best possible persona by analyzing and combining the following persona profiles generated by different AI providers.

Start your response immediately with the persona content using proper Markdown formatting. Do NOT include any preambles or meta-commentary.

INPUT PERSONAS TO COMBINE:

{{PERSONAS}}

Create a comprehensive, unified persona that represents the best synthesis of all the above profiles. Start directly with a persona title and content.`
}

func (g *Generator) formatFeedback(feedback []string) string {
	if len(feedback) == 0 {
		return "No specific feedback provided."
	}

	var formatted string
	for i, comment := range feedback {
		formatted += fmt.Sprintf("%d. %s\n", i+1, comment)
	}
	return formatted
}

func (g *Generator) combinePersonasWithFeedback(ctx context.Context, responses []ProviderResponse, feedback []string) (string, error) {
	g.logger.Info("Combining personas with feedback using Gemini")

	// Filter successful responses
	var successfulResponses []ProviderResponse
	for _, resp := range responses {
		if resp.Error == nil {
			successfulResponses = append(successfulResponses, resp)
		}
	}

	if len(successfulResponses) == 0 {
		return "", fmt.Errorf("no successful persona responses to combine")
	}

	if len(successfulResponses) == 1 {
		g.logger.Info("Only one successful response, using it directly")
		return successfulResponses[0].Content, nil
	}

	// Load feedback combination prompt template
	feedbackPrompt, err := g.loadFeedbackCombinationPrompt()
	if err != nil {
		g.logger.Warnf("Failed to load feedback combination prompt, using fallback: %v", err)
		feedbackPrompt = g.getDefaultFeedbackCombinationPrompt()
	}

	// Replace feedback placeholder
	feedbackSection := g.formatFeedback(feedback)
	feedbackPrompt = strings.Replace(feedbackPrompt, "{{FEEDBACK}}", feedbackSection, 1)

	// Build the personas section
	var personaCount int
	var personas []string
	for _, resp := range successfulResponses {
		if resp.Content == "" {
			continue
		}
		personaCount++
		personas = append(personas, fmt.Sprintf("Persona %d: %s\n<<<\n%s\n>>>", personaCount, resp.Provider, resp.Content))
	}

	finalPrompt := strings.ReplaceAll(feedbackPrompt, "{{PERSONAS}}", strings.Join(personas, "\n\n"))

	// Use Gemini with lower temperature for synthesis
	combinedPersona, err := g.gemini.GenerateSynthesis(ctx, finalPrompt)
	if err != nil {
		g.logger.Warnf("Failed to combine with Gemini, using best individual response: %v", err)
		// Fallback to the longest response as it's likely most complete
		bestResponse := successfulResponses[0]
		for _, resp := range successfulResponses[1:] {
			if len(resp.Content) > len(bestResponse.Content) {
				bestResponse = resp
			}
		}
		return bestResponse.Content, nil
	}

	return combinedPersona, nil
}

func (g *Generator) loadFeedbackCombinationPrompt() (string, error) {
	data, err := os.ReadFile("prompts/persona_combination_feedback.txt")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (g *Generator) getDefaultFeedbackCombinationPrompt() string {
	return `You are tasked with creating an improved persona by analyzing and combining the following regenerated persona profiles. Each profile was created after incorporating user feedback.

Start your response immediately with the persona content using proper Markdown formatting. Do NOT include any preambles or meta-commentary.

USER FEEDBACK TO ADDRESS:

{{FEEDBACK}}

INPUT PERSONAS TO COMBINE:

{{PERSONAS}}

Create a comprehensive, unified persona that represents the best synthesis of all the above profiles while ensuring all feedback points are addressed. Start directly with a persona title and content.`
}

func getStringValue(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
