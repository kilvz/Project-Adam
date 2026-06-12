package parser

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/google/go-github/v57/github"
)

// ParsedIssue represents the extracted information from a persona creation issue
type ParsedIssue struct {
	FullName        string
	DetailedContent string
	UserPersona     string // Optional user-supplied persona
	RawContent      string
}

// ParsePersonaIssue extracts structured information from a GitHub issue
func ParsePersonaIssue(issue *github.Issue) (*ParsedIssue, error) {
	if issue.Title == nil {
		return nil, fmt.Errorf("issue title is missing")
	}

	// Extract full name from title
	fullName, err := extractFullNameFromTitle(*issue.Title)
	if err != nil {
		return nil, fmt.Errorf("failed to extract persona name from title: %w", err)
	}

	// Handle optional body
	var detailedContent string
	var userPersona string
	var rawContent string

	if issue.Body != nil && *issue.Body != "" {
		rawContent = *issue.Body
		// Try to extract content between <<< >>>
		content, err := extractDetailedContent(*issue.Body)
		if err != nil {
			// If no markers found, use the entire body as detailed content
			detailedContent = strings.TrimSpace(*issue.Body)
		} else {
			detailedContent = content
		}

		// Try to extract user-supplied persona between [[[ ]]]
		userContent, _ := extractUserPersona(*issue.Body)
		userPersona = userContent
	}

	return &ParsedIssue{
		FullName:        fullName,
		DetailedContent: detailedContent,
		UserPersona:     userPersona,
		RawContent:      rawContent,
	}, nil
}

// extractFullNameFromTitle extracts the persona name from the issue title
func extractFullNameFromTitle(title string) (string, error) {
	// Match pattern "Create Persona: [NAME]" (case insensitive)
	re := regexp.MustCompile(`(?i)create\s+persona:\s*(.+)`)
	matches := re.FindStringSubmatch(title)

	if len(matches) < 2 {
		return "", fmt.Errorf("title must follow format 'Create Persona: [Full Name]'")
	}

	name := strings.TrimSpace(matches[1])
	if name == "" {
		return "", fmt.Errorf("persona name cannot be empty")
	}

	return name, nil
}

// extractDetailedContent extracts content between <<< and >>>
func extractDetailedContent(body string) (string, error) {
	// Find content between <<< and >>>
	startMarker := "<<<"
	endMarker := ">>>"

	startIndex := strings.Index(body, startMarker)
	if startIndex == -1 {
		return "", fmt.Errorf("missing opening marker '<<<'")
	}

	endIndex := strings.LastIndex(body, endMarker)
	if endIndex == -1 {
		return "", fmt.Errorf("missing closing marker '>>>'")
	}

	if endIndex <= startIndex {
		return "", fmt.Errorf("closing marker '>>>' must come after opening marker '<<<'")
	}

	// Extract content between markers
	content := body[startIndex+len(startMarker) : endIndex]
	content = strings.TrimSpace(content)

	if content == "" {
		return "", fmt.Errorf("no content found between markers")
	}

	// Validate that content has some structure
	if !strings.Contains(content, "**") && !strings.Contains(content, ":") {
		return "", fmt.Errorf("content appears to be unstructured - please use the template format")
	}

	return content, nil
}

// extractUserPersona extracts content between [[[ and ]]]
func extractUserPersona(body string) (string, error) {
	// Find content between [[[ and ]]]
	startMarker := "[[["
	endMarker := "]]]"

	startIndex := strings.Index(body, startMarker)
	if startIndex == -1 {
		return "", fmt.Errorf("no user persona found")
	}

	endIndex := strings.LastIndex(body, endMarker)
	if endIndex == -1 {
		return "", fmt.Errorf("missing closing marker ']]]'")
	}

	if endIndex <= startIndex {
		return "", fmt.Errorf("closing marker ']]]' must come after opening marker '[[['")
	}

	// Extract content between markers
	content := body[startIndex+len(startMarker) : endIndex]
	content = strings.TrimSpace(content)

	if content == "" || content == "[Paste your complete persona here]" {
		return "", fmt.Errorf("no user persona content provided")
	}

	return content, nil
}

// FormatForPrompt formats the parsed issue content for AI generation
func (p *ParsedIssue) FormatForPrompt() string {
	if p.DetailedContent == "" {
		// No additional content provided, create persona based on name only
		return fmt.Sprintf(`Create a comprehensive persona for: %s

No additional details were provided. Please create a detailed persona profile based on your knowledge of this person/character, including their background, personality traits, speaking style, areas of expertise, values, beliefs, goals, and any other relevant characteristics that would help in AI interactions.`,
			p.FullName)
	}

	return fmt.Sprintf(`Create a comprehensive persona for: %s

%s

Please create a detailed persona profile that captures all the provided information and expands upon it to create a complete, nuanced character that can be used for AI interactions.`,
		p.FullName, p.DetailedContent)
}

// GetParsingErrorComment generates a helpful error comment for the issue
func GetParsingErrorComment(err error) string {
	errorDetail := err.Error()
	return fmt.Sprintf(`❌ **Unable to Parse Issue**

I couldn't parse this issue to create a persona. %s

**Required Format:**
1. **Title**: Must be "Create Persona: [Full Name]"
2. **Body**: Must include content wrapped in <<< and >>> markers

**Example:**
`+"```markdown"+`
**Full Name:** David Attenborough

<<<
**Background & Context:**
Natural historian and broadcaster...

**Personality Traits:**
- Gentle and patient
- Deeply curious...

[Continue with other sections]
>>>
`+"```"+`

Please update your issue to match the template format and I'll process it. You can find the full template in the repository's issue templates.

---
*[Studio](https://github.com/twin2ai/studio) - Multi-AI Persona Generation Pipeline*`, errorDetail)
}
