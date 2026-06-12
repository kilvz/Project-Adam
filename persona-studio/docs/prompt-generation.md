# Prompt Generation System

## Overview

The Studio prompt generation system automatically creates platform-specific and variation prompts based on persona `synthesized.md` files. When a synthesized persona is updated, the system generates optimized prompts for different AI platforms (ChatGPT, Claude, Gemini, Discord Bot, Character.AI) and creates condensed variations for immediate use.

## How It Works

### 1. Trigger Detection

The system detects when prompt generation is needed through:

- **File Modification**: Monitors when `synthesized.md` is newer than existing prompts
- **README Markers**: Detects HTML comment triggers in persona READMEs
- **Manual Triggers**: Programmatic or CLI-based prompt generation requests
- **Asset System Integration**: Works with the existing asset monitoring system

### 2. Template-Based Generation

Each prompt type uses a specialized template:

#### Platform-Specific Prompts
- **ChatGPT** (`prompts/platform_chatgpt.txt`): Optimized for ChatGPT's conversational interface
- **Claude** (`prompts/platform_claude.txt`): Tailored for Claude's analytical and ethical reasoning
- **Gemini** (`prompts/platform_gemini.txt`): Designed for Gemini's multimodal capabilities
- **Discord Bot** (`prompts/platform_discord.txt`): Casual personality for Discord communities
- **Character.AI** (`prompts/platform_characterai.txt`): Immersive character for roleplay

#### Variation Prompts
- **Condensed** (`prompts/variation_condensed.txt`): Compact 300-600 word version for quick use
- **Alternative** (`prompts/variation_alternative.txt`): Different facets and emphasis areas

### 3. AI-Powered Generation

All prompts are generated using **Gemini 2.0 Flash** with:
- **Temperature**: 0.3 (for consistent, reliable prompt generation)
- **Max Tokens**: 4000 (for detailed, comprehensive prompts)
- **Optimized Settings**: Configured for prompt generation tasks

### 4. Repository Integration

Generated prompts are automatically saved to the persona repository:

```
personas/
├── david_attenborough/
│   ├── synthesized.md              # Source content
│   ├── README.md                   # Updated with prompt links
│   ├── .assets_status.json         # Updated generation status
│   └── prompts/                    # Generated prompts directory
│       ├── chatgpt.md              # ChatGPT system prompt
│       ├── claude.md               # Claude system prompt
│       ├── gemini.md               # Gemini system prompt
│       ├── discord.md              # Discord bot personality
│       ├── characterai.md          # Character.AI character
│       ├── condensed.md            # Prompt-ready version
│       └── alternative.md          # Alternative variations
```

## Usage

### Automatic Generation

Add trigger markers to persona READMEs:

```html
<!-- To generate all prompts -->
<!-- GENERATE:prompts -->

<!-- To generate only platform-specific prompts -->
<!-- GENERATE:platform_prompts -->

<!-- To generate only variation prompts -->
<!-- GENERATE:variation_prompts -->
```

### Command Line Tool

```bash
# Generate all prompts for a specific persona
go run cmd/prompt-generator/main.go -persona "David Attenborough"

# Generate only platform prompts
go run cmd/prompt-generator/main.go -persona "David Attenborough" -type platform

# Generate only variation prompts
go run cmd/prompt-generator/main.go -persona "David Attenborough" -type variation

# Force regeneration even if prompts exist
go run cmd/prompt-generator/main.go -persona "David Attenborough" -force

# Show statistics
go run cmd/prompt-generator/main.go -persona "David Attenborough" -stats

# Run continuous monitoring
go run cmd/prompt-generator/main.go -interval 5m

# Run once for all personas
go run cmd/prompt-generator/main.go -once

# Debug mode
go run cmd/prompt-generator/main.go -log debug -persona "David Attenborough"
```

### Programmatic Usage

```go
import (
    "github.com/twin2ai/studio/internal/prompts"
    "github.com/twin2ai/studio/internal/gemini"
)

// Create prompt service
geminiClient := gemini.NewClient(apiKey, model, logger)
promptService := prompts.NewService(geminiClient, logger, baseDir)

// Generate all prompts
ctx := context.Background()
err := promptService.TriggerPromptGeneration(ctx, "David Attenborough")

// Check if prompts are needed
needs, err := promptService.CheckPersonaNeedsPrompts("David Attenborough")

// Get generation statistics
stats, err := promptService.GetPromptGenerationStats("David Attenborough")
```

## Prompt Templates

### Template Structure

Each template follows this pattern:

```
You are tasked with creating a [Platform]-optimized prompt based on a comprehensive persona.

INPUT PERSONA:
{{SYNTHESIZED_PERSONA}}

INSTRUCTIONS:
Create a [Platform] system prompt that:
1. [Platform-specific requirement 1]
2. [Platform-specific requirement 2]
...

FORMAT REQUIREMENTS:
- [Platform-specific formatting]
- [Optimization notes]
...

Create a [description] that [goals].
```

### Template Customization

Templates can be customized by editing files in the `prompts/` directory:

- `platform_chatgpt.txt`: ChatGPT optimization
- `platform_claude.txt`: Claude optimization
- `platform_gemini.txt`: Gemini optimization
- `platform_discord.txt`: Discord bot personality
- `platform_characterai.txt`: Character.AI character
- `variation_condensed.txt`: Condensed version
- `variation_alternative.txt`: Alternative variations

## Generated Prompt Format

Each generated prompt includes:

```markdown
# [Prompt Display Name]

> **Generated:** 2025-07-13T10:30:00Z  
> **Persona:** David Attenborough  
> **Type:** chatgpt  
> **Source:** synthesized.md  

---

[Generated prompt content goes here]

---

*Generated automatically by Studio using Gemini 2.0 Flash*  
*Last updated: 2025-07-13 10:30:00 UTC*
```

## Integration with Asset System

The prompt generation system integrates with Studio's asset monitoring:

### Asset Types
- `prompts`: All prompt generation
- `platform_prompts`: Platform-specific prompts only
- `variation_prompts`: Variation prompts only

### Status Tracking
- **Pending**: Prompts marked for generation
- **Generated**: Successfully created prompts
- **Failed**: Generation errors tracked
- **Timestamps**: Last generation times recorded

### Automatic Updates
- **README**: Updated with prompt links and descriptions
- **Asset Status**: Generation flags and timestamps updated
- **File Management**: Old prompts replaced, new ones added

## Quality Assurance

### Validation
- **Minimum Length**: 100 characters minimum
- **Template Placeholders**: Ensures all `{{}}` markers are replaced
- **Content Validation**: Checks for empty or invalid content
- **Format Verification**: Validates prompt structure and metadata

### Error Handling
- **Graceful Failures**: Individual prompt failures don't stop the batch
- **Retry Logic**: Failed generations can be retried
- **Status Tracking**: Errors logged and tracked in asset status
- **Fallback Behavior**: System continues with successful generations

## Best Practices

### Template Development
1. **Platform Research**: Understand each platform's optimal prompt format
2. **Testing**: Test generated prompts with actual AI platforms
3. **Iteration**: Refine templates based on generation quality
4. **Documentation**: Update template documentation with changes

### Generation Management
1. **Monitoring**: Set up continuous monitoring for automatic updates
2. **Validation**: Always validate generated prompts before use
3. **Backup**: Keep copies of working prompts before regeneration
4. **Version Control**: Track prompt changes over time

### Performance Optimization
1. **Batch Generation**: Generate multiple prompts together when possible
2. **Rate Limiting**: Respect Gemini API rate limits
3. **Caching**: Avoid regenerating unchanged prompts
4. **Resource Management**: Monitor API usage and costs

## Troubleshooting

### Common Issues

**Templates Not Found**
- Ensure template files exist in `prompts/` directory
- Check file naming conventions match prompt types

**Generation Failures**
- Verify Gemini API key and model access
- Check synthesized.md file exists and is readable
- Review API rate limits and quotas

**Invalid Prompts**
- Check template placeholder replacement
- Verify minimum length requirements
- Review content for completeness

**File Permissions**
- Ensure write permissions to persona directories
- Check directory creation permissions
- Verify file overwrite permissions

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
go run cmd/prompt-generator/main.go -log debug -persona "David Attenborough"
```

This provides:
- Template loading details
- API request/response information
- File operation logging
- Validation step details

## Future Enhancements

### Planned Features
- **Multi-model Generation**: Support for multiple AI providers
- **A/B Testing**: Generate and compare different prompt variations
- **Quality Scoring**: Automatic quality assessment of generated prompts
- **Template Marketplace**: Shared template repository
- **Performance Analytics**: Track prompt effectiveness metrics

### Integration Opportunities
- **CI/CD Pipelines**: Automatic prompt regeneration on persona updates
- **Webhook Integration**: Real-time prompt updates via webhooks
- **External APIs**: Integration with prompt management platforms
- **Version Control**: Git-based prompt version tracking

## Configuration

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash-exp

# Optional
LOG_LEVEL=info
PROMPT_GENERATION_ENABLED=true
PROMPT_GENERATION_INTERVAL=5m
```

### Custom Configuration

```json
{
  "prompt_generation": {
    "enabled": true,
    "auto_generate": true,
    "templates_dir": "prompts",
    "output_format": "markdown",
    "validation": {
      "min_length": 100,
      "check_placeholders": true
    },
    "gemini": {
      "model": "gemini-2.0-flash-exp",
      "temperature": 0.3,
      "max_tokens": 4000
    }
  }
}
```