package multiprovider

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/google/go-github/v57/github"
	"github.com/twin2ai/studio/internal/assets"
	gh "github.com/twin2ai/studio/internal/github"
	"github.com/twin2ai/studio/pkg/models"
)

// ProcessIssueWithStructure processes an issue and generates a complete persona package
func (g *Generator) ProcessIssueWithStructure(ctx context.Context, issue *github.Issue) (*models.Persona, *gh.PersonaFiles, error) {
	return g.ProcessIssueWithStructureAndUser(ctx, issue, "")
}

// ProcessIssueWithStructureAndUser processes an issue with optional user-supplied persona
func (g *Generator) ProcessIssueWithStructureAndUser(ctx context.Context, issue *github.Issue, userPersona string) (*models.Persona, *gh.PersonaFiles, error) {
	g.logger.Infof("Processing issue #%d with structured multi-provider generation: %s", *issue.Number, *issue.Title)
	if userPersona != "" {
		g.logger.Info("User-supplied persona detected, will include in synthesis")
	}

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
		return nil, nil, fmt.Errorf("failed to generate personas from providers: %w", err)
	}

	// Store individual responses
	if err := g.storeArtifacts(*issue.Number, responses); err != nil {
		g.logger.Warnf("Failed to store artifacts: %v", err)
	}

	// Extract individual provider contents
	var claudeRaw, geminiRaw, grokRaw, gptRaw string
	for _, resp := range responses {
		if resp.Error == nil {
			switch resp.Provider {
			case "claude":
				claudeRaw = resp.Content
			case "gemini":
				geminiRaw = resp.Content
			case "grok":
				grokRaw = resp.Content
			case "gpt":
				gptRaw = resp.Content
			}
		}
	}

	// Combine all responses into final persona (including user persona if provided)
	fullSynthesis, err := g.combinePersonasWithUser(ctx, responses, userPersona)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to combine personas: %w", err)
	}

	// Store combined result
	if err := g.storeCombinedPersona(*issue.Number, fullSynthesis); err != nil {
		g.logger.Warnf("Failed to store combined persona: %v", err)
	}

	// Use persona name from issue title
	personaName := *issue.Title

	// Create asset status
	assetStatus := &assets.AssetStatus{
		PersonaName:           personaName,
		LastSynthesizedUpdate: time.Now(),
		LastAssetsGeneration:  time.Time{}, // Zero time means never generated
		PendingAssets:         []string{},
		GeneratedAssets:       []string{},
		AssetGenerationFlags:  make(map[string]bool),
		Metadata:              make(map[string]string),
	}

	// Create PersonaFiles structure
	files := &gh.PersonaFiles{
		ClaudeRaw:     claudeRaw,
		GeminiRaw:     geminiRaw,
		GrokRaw:       grokRaw,
		GPTRaw:        gptRaw,
		UserRaw:       userPersona,
		FullSynthesis: fullSynthesis,
		AssetStatus:   assetStatus,
	}

	// Create Persona model
	persona := &models.Persona{
		Name:        personaName,
		Content:     fullSynthesis,
		IssueNumber: *issue.Number,
	}

	return persona, files, nil
}

// loadPromptFromFile loads a prompt template from a file
func (g *Generator) loadPromptFromFile(filename string) (string, error) {
	data, err := os.ReadFile(filepath.Join("prompts", filename))
	if err != nil {
		return "", fmt.Errorf("failed to load prompt from %s: %w", filename, err)
	}
	return string(data), nil
}

// UpdatePersonaWithUserInput updates an existing persona with user-provided content
func (g *Generator) UpdatePersonaWithUserInput(ctx context.Context, personaName string, existingPersona string, userPersona string) (string, error) {
	g.logger.Infof("Updating persona '%s' with user input", personaName)

	// Combine existing and user personas for synthesis
	combinationPrompt := fmt.Sprintf(`You are tasked with creating an improved persona by synthesizing an existing persona with a user-provided update.

The user has provided their own version of the persona that should be intelligently merged with the existing version to create a superior, comprehensive persona that incorporates the best elements of both.

EXISTING PERSONA:
<<<
%s
>>>

USER-PROVIDED UPDATE:
<<<
%s
>>>

Please create a synthesized version that:
1. Incorporates new information from the user's version
2. Preserves valuable details from the existing persona
3. Resolves any conflicts by preferring the user's interpretation
4. Maintains consistency and coherence throughout
5. Results in a richer, more complete persona

Start your response immediately with the synthesized persona content using proper Markdown formatting. Do NOT include any preambles or meta-commentary.`, existingPersona, userPersona)

	// Use Gemini with lower temperature for synthesis
	synthesized, err := g.gemini.GenerateSynthesis(ctx, combinationPrompt)
	if err != nil {
		return "", fmt.Errorf("failed to synthesize personas: %w", err)
	}

	return synthesized, nil
}

// RegeneratePersonaWithStructuredFeedback regenerates a complete persona package with feedback
func (g *Generator) RegeneratePersonaWithStructuredFeedback(ctx context.Context, issue *github.Issue, existingPersona string, feedback []string) (*models.Persona, *gh.PersonaFiles, error) {
	g.logger.Infof("Regenerating structured persona for issue #%d with feedback", *issue.Number)

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
		return nil, nil, fmt.Errorf("failed to regenerate personas from providers: %w", err)
	}

	// Store individual responses with feedback suffix
	if err := g.storeArtifactsWithSuffix(*issue.Number, responses, "feedback"); err != nil {
		g.logger.Warnf("Failed to store feedback artifacts: %v", err)
	}

	// Continue with the same structure generation as ProcessIssueWithStructure
	// ... (rest of the implementation follows the same pattern)

	// Extract individual provider contents
	var claudeRaw, geminiRaw, grokRaw, gptRaw string
	for _, resp := range responses {
		if resp.Error == nil {
			switch resp.Provider {
			case "claude":
				claudeRaw = resp.Content
			case "gemini":
				geminiRaw = resp.Content
			case "grok":
				grokRaw = resp.Content
			case "gpt":
				gptRaw = resp.Content
			}
		}
	}

	// Combine all responses into final persona
	fullSynthesis, err := g.combinePersonas(ctx, responses)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to combine feedback personas: %w", err)
	}

	// Store combined result with feedback suffix
	if err := g.storeCombinedPersonaWithSuffix(*issue.Number, fullSynthesis, "feedback"); err != nil {
		g.logger.Warnf("Failed to store combined feedback persona: %v", err)
	}

	// Use persona name from issue title
	personaName := *issue.Title

	// Create asset status for regenerated persona
	assetStatus := &assets.AssetStatus{
		PersonaName:           personaName,
		LastSynthesizedUpdate: time.Now(),
		LastAssetsGeneration:  time.Time{}, // Zero time means never generated
		PendingAssets:         []string{},
		GeneratedAssets:       []string{},
		AssetGenerationFlags:  make(map[string]bool),
		Metadata:              map[string]string{"regenerated": "true"},
	}

	// Create PersonaFiles structure
	files := &gh.PersonaFiles{
		ClaudeRaw:     claudeRaw,
		GeminiRaw:     geminiRaw,
		GrokRaw:       grokRaw,
		GPTRaw:        gptRaw,
		UserRaw:       "", // No user persona in regeneration
		FullSynthesis: fullSynthesis,
		AssetStatus:   assetStatus,
	}

	// Create Persona model
	persona := &models.Persona{
		Name:        personaName,
		Content:     fullSynthesis,
		IssueNumber: *issue.Number,
	}

	return persona, files, nil
}
