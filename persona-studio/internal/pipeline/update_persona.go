package pipeline

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/go-github/v57/github"
)

// UpdatePersonaRequest represents a request to update an existing persona
type UpdatePersonaRequest struct {
	PersonaName string
	UserPersona string
}

// ProcessPersonaUpdate handles updating an existing persona with user input
func (p *Pipeline) ProcessPersonaUpdate(ctx context.Context, request UpdatePersonaRequest) error {
	p.logger.Infof("Processing persona update for: %s", request.PersonaName)

	// Retrieve existing persona from repository
	existingPersona, err := p.github.GetExistingPersona(ctx, request.PersonaName)
	if err != nil {
		return fmt.Errorf("failed to retrieve existing persona: %w", err)
	}

	p.logger.Info("Retrieved existing persona from repository")

	// Synthesize with user input using Gemini
	synthesizedPersona, err := p.multiGenerator.UpdatePersonaWithUserInput(
		ctx,
		request.PersonaName,
		existingPersona,
		request.UserPersona,
	)
	if err != nil {
		return fmt.Errorf("failed to synthesize persona update: %w", err)
	}

	p.logger.Info("Successfully synthesized user input with existing persona")

	// Create pull request with updated persona
	pr, err := p.github.UpdatePersonaWithUserInput(
		ctx,
		request.PersonaName,
		existingPersona,
		request.UserPersona,
		synthesizedPersona,
	)
	if err != nil {
		return fmt.Errorf("failed to create update PR: %w", err)
	}

	p.logger.Infof("Created update PR #%d for persona: %s", *pr.Number, request.PersonaName)

	return nil
}

// ParseUpdateRequest parses an update request from a GitHub issue
func ParseUpdateRequest(issue *github.Issue) (*UpdatePersonaRequest, error) {
	if issue.Title == nil {
		return nil, fmt.Errorf("issue title is missing")
	}

	// Expected format: "Update Persona: [Name]"
	title := *issue.Title
	if !strings.HasPrefix(strings.ToLower(title), "update persona:") {
		return nil, fmt.Errorf("title must start with 'Update Persona:'")
	}

	// Extract persona name
	parts := strings.SplitN(title, ":", 2)
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid title format")
	}

	personaName := strings.TrimSpace(parts[1])
	if personaName == "" {
		return nil, fmt.Errorf("persona name cannot be empty")
	}

	// Extract user persona from body
	if issue.Body == nil || *issue.Body == "" {
		return nil, fmt.Errorf("issue body must contain the updated persona")
	}

	// Look for content between [[[ ]]] markers
	body := *issue.Body
	startMarker := "[[["
	endMarker := "]]]"

	startIndex := strings.Index(body, startMarker)
	endIndex := strings.LastIndex(body, endMarker)

	var userPersona string
	if startIndex != -1 && endIndex != -1 && endIndex > startIndex {
		// Extract content between markers
		userPersona = body[startIndex+len(startMarker) : endIndex]
		userPersona = strings.TrimSpace(userPersona)
	} else {
		// Use entire body if no markers
		userPersona = strings.TrimSpace(body)
	}

	if userPersona == "" {
		return nil, fmt.Errorf("no persona content provided")
	}

	return &UpdatePersonaRequest{
		PersonaName: personaName,
		UserPersona: userPersona,
	}, nil
}
