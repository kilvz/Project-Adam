# Structured Persona Generation Pipeline

## Overview

The Studio pipeline has been enhanced to create a comprehensive persona package with multiple versions and formats, organized in a structured folder hierarchy.

## New Folder Structure

Instead of a flat `personas/persona_name.md` file, each persona now gets its own folder:

```
personas/
├── david_attenborough/
│   ├── README.md                    # Persona overview and usage guide
│   ├── raw/                        # Individual AI provider outputs
│   │   ├── claude.md
│   │   ├── gemini.md
│   │   ├── grok.md
│   │   └── gpt.md
│   └── synthesized.md              # Full combined persona
```

## File Descriptions

### 1. **Raw Outputs** (`/raw/`)
- Contains the original, unmodified outputs from each AI provider
- Useful for analysis and understanding different AI interpretations
- Files: `claude.md`, `gemini.md`, `grok.md`, `gpt.md`

### 2. **Synthesized Version** (`synthesized.md`)
- The complete, combined persona using Gemini's synthesis
- Incorporates the best elements from all four AI providers
- Most comprehensive version for detailed use

### 3. **Prompt-Ready Version** (`prompt_ready.md`)
- Condensed 500-1000 word version
- Optimized for immediate use in AI prompts
- Captures essential characteristics and behaviors
- Best for quick implementation

- Gemini
- Character.AI
- Discord Bot
- Twitter/X
- LinkedIn
- Email Assistant

## Implementation Details

### New Components

1. **`internal/github/structured_pr.go`**
   - `CreateStructuredPersonaPR()`: Creates PRs with multiple files
   - `PersonaFiles` struct: Holds all persona variations
   - `generatePersonaReadme()`: Creates folder README

2. **`internal/multiprovider/enhanced_generator.go`**
   - `ProcessIssueWithStructure()`: Generates complete persona package
   - `generatePromptReadyVersion()`: Creates condensed version

3. **`internal/pipeline/structured_pipeline.go`**
   - `processNewIssuesWithStructure()`: Handles new issues
   - `processPRCommentsWithStructure()`: Updates all files on feedback
   - `updateStructuredPR()`: Updates entire persona package

### Pipeline Flow

1. **Issue Processing**:
   - Issue with `create-persona` label detected
   - All 4 AI providers generate personas in parallel
   - Gemini synthesizes into full version
   - Additional versions generated (prompt-ready)
   - Complete package created in structured folders

2. **PR Creation**:
   - Creates folder structure with all files
   - Includes comprehensive README
   - Labels: `persona`, `automated`, `studio`, `structured`

3. **Feedback Processing**:
   - Comments trigger regeneration of entire package
   - All files updated with improved versions
   - Maintains consistency across all formats

## Benefits

1. **Organization**: Clear folder structure for each persona
2. **Flexibility**: Multiple versions for different use cases
3. **Discoverability**: README guides users to right version
4. **Analysis**: Raw outputs preserved for comparison
5. **Immediate Use**: Synthesized version ready for use

## Usage Recommendations

- **Quick Implementation**: Use `prompt_ready.md`
- **Detailed Work**: Use `synthesized.md`
- **Analysis**: Compare `/raw/` outputs

## Future Enhancements

- Version history tracking
- Persona evolution over time
- Cross-persona analysis
- Automated testing of personas
- Usage analytics per version