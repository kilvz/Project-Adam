# Issue Template and Parsing System

## Overview

The Studio pipeline now uses a standardized issue template that enables consistent parsing and high-quality persona generation. Issues that don't match the format receive helpful error comments.

## Template Format

### Required Structure

```markdown
Title: Create Persona: [FULL NAME]

Body: (OPTIONAL - can be empty)
```

### Optional Structured Body

If you want to provide additional details:

```markdown
**Full Name:** [Full name of persona]

<<<
[Structured content with sections]
>>>

[Optional reference materials outside markers]
```

### Key Components

1. **Title Pattern**: `Create Persona: [Full Name]`
   - Case insensitive matching
   - Full name extracted after the colon

2. **Content Markers**: `<<<` and `>>>`
   - All persona details must be between these markers
   - Content outside markers is ignored for generation

3. **Structured Sections** (inside markers):
   - Background & Context
   - Personality Traits
   - Speaking Style & Voice
   - Areas of Expertise
   - Values & Beliefs
   - Goals & Motivations
   - Additional Details

## Implementation Details

### Parser Module (`internal/parser/issue_parser.go`)

```go
type ParsedIssue struct {
    FullName        string  // Extracted from title
    DetailedContent string  // Content between markers
    RawContent      string  // Original issue body
}
```

#### Key Functions:

1. **ParsePersonaIssue()**: Main parsing function
   - Validates title format
   - Extracts content between markers
   - Returns structured data or error

2. **GetParsingErrorComment()**: Generates helpful error messages
   - Explains what went wrong
   - Shows correct format
   - Provides examples

### Pipeline Integration

The pipeline now:
1. Attempts to parse each new issue
2. On success: Processes with enhanced content
3. On failure: 
   - Comments with specific error and examples
   - Marks issue as processed to prevent spam
   - Logs parsing failure

### Error Handling

Common parsing errors handled:
- Missing or incorrect title format
- Missing opening `<<<` marker
- Missing closing `>>>` marker
- Empty content between markers
- Unstructured content (no sections)

Each error generates a specific, helpful message.

## Benefits

1. **Consistency**: All personas follow same structure
2. **Quality**: Structured input leads to better output
3. **User-Friendly**: Clear errors help users fix issues
4. **Parsing Safety**: Invalid issues don't crash pipeline
5. **Flexibility**: Content outside markers for references

## Example Parsing Flow

### Valid Issue:
```
Title: "Create Persona: Marie Curie"
Body: Contains properly formatted content with markers
Result: Successfully parsed and processed
```

### Invalid Issue:
```
Title: "Make a persona for Einstein"
Result: Error comment explaining title format
```

### Missing Markers:
```
Title: "Create Persona: Einstein"
Body: Content without <<< >>> markers
Result: Error comment showing marker usage
```

## Testing

The parser can be tested with:
```bash
go run examples/test_parser.go
```

This demonstrates:
- Successful parsing
- Title format errors
- Missing marker errors
- Generated error comments