# Prompt Templates

## Overview

Studio uses external prompt template files for persona synthesis. This allows for easy customization without modifying code.

## Template Files

All prompt templates are stored in the `prompts/` directory:

### 1. `persona_combination.txt`
Combines multiple AI-generated personas into a single synthesized version.

**Input**: Multiple personas wrapped in delimiters
**Output**: Unified synthesized persona

### 2. `persona_combination_feedback.txt`
Regenerates personas based on user feedback.

**Input**: 
- `{{FEEDBACK}}` - User feedback
- `{{PERSONAS}}` - Regenerated personas
**Output**: Improved synthesized persona

## Data Flow

```
1. AI Providers → Individual Personas
2. Synthesis → Synthesized Persona
```

The synthesized persona is the single source for all variations, ensuring consistency.

## Customization

To customize the generation process:

1. Edit the relevant prompt file in `prompts/`
2. Use the placeholder variables:
   - `{{PERSONAS}}` - The persona content to process
   - `{{FEEDBACK}}` - User feedback (for regeneration)
3. Save the file - changes take effect immediately

## Template Structure

### Placeholders
- `{{SYNTHESIZED_PERSONA}}`: Replaced with the full synthesized persona
- `{{PLATFORM}}`: Replaced with the platform name (ChatGPT, Claude, etc.)

### Best Practices
1. Start with clear instructions
2. Use numbered lists for structured output
3. Include "Start immediately with..." to avoid preambles
4. Be specific about constraints (word counts, format)

## Benefits

1. **Flexibility**: Modify prompts without code changes
2. **Consistency**: All variations derive from synthesized version
3. **Transparency**: Prompts are visible and editable
4. **Version Control**: Track prompt changes in git
5. **Experimentation**: Easy to test different approaches

## Fallback Behavior

If a prompt file is missing or unreadable:
1. A warning is logged
2. A hardcoded fallback prompt is used
3. Generation continues normally

This ensures the pipeline never fails due to missing prompt files.