package main

import (
	"fmt"
	"log"

	"github.com/google/go-github/v57/github"
	"github.com/twin2ai/studio/internal/parser"
)

func main() {
	// Example 1: Valid issue
	validTitle := "Create Persona: David Attenborough"
	validBody := `**Full Name:** David Attenborough

<<<
**Background & Context:**
Natural historian, broadcaster, and environmental advocate. Known for nature documentaries.

**Personality Traits:**
- Gentle and patient
- Deeply curious about nature
- Passionate educator

**Speaking Style & Voice:**
- Soft, measured tone
- Uses poetic descriptions
- Often whispers to avoid disturbing wildlife

**Areas of Expertise:**
- Natural history
- Wildlife behavior
- Environmental conservation

**Values & Beliefs:**
- Respect for all life
- Environmental stewardship
- Education through wonder

**Goals & Motivations:**
- Share the beauty of nature
- Inspire conservation
- Document disappearing worlds

**Additional Details:**
- Over 70 years in broadcasting
- Knighted by Queen Elizabeth II
- Narrated Planet Earth series
>>>

**Reference Materials:**
- Planet Earth documentary series
- Life on Earth book series`

	issue1 := &github.Issue{
		Title: &validTitle,
		Body:  &validBody,
	}

	// Parse valid issue
	fmt.Println("=== Testing Valid Issue ===")
	parsed, err := parser.ParsePersonaIssue(issue1)
	if err != nil {
		log.Printf("Error: %v", err)
		fmt.Println("\nError Comment:")
		fmt.Println(parser.GetParsingErrorComment(err))
	} else {
		fmt.Printf("✅ Successfully parsed!\n")
		fmt.Printf("Full Name: %s\n", parsed.FullName)
		fmt.Printf("Detailed Content Length: %d characters\n", len(parsed.DetailedContent))
		fmt.Printf("\nFormatted for prompt:\n%s\n", parsed.FormatForPrompt())
	}

	fmt.Println("\n" + strings.Repeat("=", 50) + "\n")

	// Example 2: Invalid title format
	invalidTitle := "David Attenborough Persona"
	issue2 := &github.Issue{
		Title: &invalidTitle,
		Body:  &validBody,
	}

	fmt.Println("=== Testing Invalid Title ===")
	_, err = parser.ParsePersonaIssue(issue2)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		fmt.Println("\nError Comment Preview:")
		fmt.Println(parser.GetParsingErrorComment(err))
	}

	fmt.Println("\n" + strings.Repeat("=", 50) + "\n")

	// Example 3: Missing markers
	missingMarkers := `**Full Name:** Sherlock Holmes

**Background & Context:**
Fictional detective created by Arthur Conan Doyle.

**Personality Traits:**
- Highly observant
- Logical thinker`

	issue3 := &github.Issue{
		Title: &validTitle,
		Body:  &missingMarkers,
	}

	fmt.Println("=== Testing Missing Markers ===")
	_, err = parser.ParsePersonaIssue(issue3)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		fmt.Println("\nWould comment on issue with:")
		fmt.Println(parser.GetParsingErrorComment(err))
	}
}

var strings = struct {
	Repeat func(string, int) string
}{
	Repeat: func(s string, n int) string {
		result := ""
		for i := 0; i < n; i++ {
			result += s
		}
		return result
	},
}
