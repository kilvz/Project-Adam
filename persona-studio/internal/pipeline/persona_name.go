package pipeline

import (
	"fmt"
	"regexp"
	"strings"
)

// PersonaName represents a parsed persona name with optional alias
type PersonaName struct {
	PrimaryName string // The main name (alias/brand name if available, otherwise real name)
	RealName    string // The real name (if different from primary)
	FullName    string // The complete formatted name for display
}

// ParsePersonaName parses various name formats and returns a structured PersonaName
func ParsePersonaName(input string) (*PersonaName, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return nil, fmt.Errorf("empty name")
	}

	// Pattern 1: "Alias (Real Name)" - e.g., "MrBeast (Jimmy Donaldson)"
	aliasPattern := regexp.MustCompile(`^([^(]+)\s*\(([^)]+)\)\s*$`)
	if matches := aliasPattern.FindStringSubmatch(input); len(matches) == 3 {
		return &PersonaName{
			PrimaryName: strings.TrimSpace(matches[1]),
			RealName:    strings.TrimSpace(matches[2]),
			FullName:    input,
		}, nil
	}

	// Pattern 2: "Real Name aka Alias" - e.g., "Jimmy Donaldson aka MrBeast"
	akaPattern := regexp.MustCompile(`^(.+?)\s+aka\s+(.+)$`)
	if matches := akaPattern.FindStringSubmatch(input); len(matches) == 3 {
		realName := strings.TrimSpace(matches[1])
		alias := strings.TrimSpace(matches[2])
		return &PersonaName{
			PrimaryName: alias,
			RealName:    realName,
			FullName:    fmt.Sprintf("%s (%s)", alias, realName),
		}, nil
	}

	// Pattern 3: Simple name (no alias)
	return &PersonaName{
		PrimaryName: input,
		RealName:    input,
		FullName:    input,
	}, nil
}

// HasAlias returns true if the persona has different primary and real names
func (pn *PersonaName) HasAlias() bool {
	return pn.PrimaryName != pn.RealName
}

// GetSearchVariations returns all variations of the name for duplicate checking
func (pn *PersonaName) GetSearchVariations() []string {
	variations := []string{pn.PrimaryName}

	if pn.HasAlias() {
		variations = append(variations, pn.RealName)
		// Add the full formatted version
		variations = append(variations, fmt.Sprintf("%s (%s)", pn.PrimaryName, pn.RealName))
		variations = append(variations, fmt.Sprintf("%s aka %s", pn.RealName, pn.PrimaryName))
	}

	// Add lowercase variations for search
	lowercaseVars := make([]string, 0, len(variations))
	for _, v := range variations {
		lowercaseVars = append(lowercaseVars, strings.ToLower(v))
	}
	variations = append(variations, lowercaseVars...)

	return variations
}

// GetPromptDescription returns a description suitable for AI prompts
func (pn *PersonaName) GetPromptDescription() string {
	if pn.HasAlias() {
		return fmt.Sprintf("%s (also known as %s)", pn.PrimaryName, pn.RealName)
	}
	return pn.PrimaryName
}

// GetDirectoryName returns a sanitized name suitable for directory creation
func (pn *PersonaName) GetDirectoryName() string {
	// Use primary name for directory
	name := strings.ToLower(pn.PrimaryName)

	// Replace spaces and special characters
	replacer := strings.NewReplacer(
		" ", "_",
		"/", "_",
		"\\", "_",
		":", "_",
		"*", "_",
		"?", "_",
		"\"", "_",
		"<", "_",
		">", "_",
		"|", "_",
		".", "_",
		",", "_",
		"'", "",
		"`", "",
		"(", "",
		")", "",
	)

	return replacer.Replace(name)
}

// GetTrackingKey returns a unique key for tracking processed names
func (pn *PersonaName) GetTrackingKey() string {
	if pn.HasAlias() {
		return fmt.Sprintf("%s|%s", pn.PrimaryName, pn.RealName)
	}
	return pn.PrimaryName
}
