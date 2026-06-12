# Prompt Generation GitHub PR Integration

## Overview

Studio now automatically creates GitHub pull requests when generating prompts for personas. When a `synthesized.md` file is updated or prompt generation is triggered, the system generates optimized prompts for different platforms and submits them as a comprehensive PR to the personas repository.

## How It Works

### 1. Automatic PR Creation

When prompt generation is triggered:

1. **Detection**: System detects `synthesized.md` updates or README trigger markers
2. **Generation**: Gemini 2.0 Flash generates 7 platform-specific and variation prompts
3. **Validation**: Each prompt is validated for quality and completeness
4. **Branch Creation**: Creates a timestamped branch like `prompts/david-attenborough-20250713-103045`
5. **File Commit**: Commits all generated prompts and updated files to the branch
6. **PR Creation**: Creates a comprehensive pull request with detailed description
7. **Issue Linking**: Comments on the original issue (if applicable) with PR link

### 2. PR Content Structure

Each PR includes:

```
personas/persona_name/
├── prompts/                    # All generated prompts
│   ├── chatgpt.md              # ChatGPT system prompt
│   ├── claude.md               # Claude system prompt
│   ├── gemini.md               # Gemini system prompt
│   ├── discord.md              # Discord bot personality
│   ├── characterai.md          # Character.AI character
│   ├── condensed.md            # Prompt-ready version
│   └── alternative.md          # Alternative variations
├── README.md                   # Updated with prompt links
└── .assets_status.json         # Updated generation status
```

### 3. PR Description Template

Generated PRs include:
- **Title**: "Update prompts for persona: [Persona Name]"
- **Platform-specific prompts list** with checkmarks
- **Prompt variations list** with checkmarks
- **Files updated section**
- **Usage instructions**
- **Generation details** (model, timestamp, source)
- **Related issue links** (if applicable)

## Configuration

### Environment Variables

Required for PR creation:
```bash
# GitHub Integration
GITHUB_TOKEN=your_github_token_with_repo_permissions
GITHUB_OWNER=your_username_or_org
GITHUB_REPO=your_issues_repo
PERSONAS_OWNER=twin2ai
PERSONAS_REPO=personas

# Gemini Integration (for prompt generation)
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash-exp
```

### Permissions Required

The GitHub token needs:
- **Repository access** to both issues and personas repositories
- **Contents: Write** permission for creating files
- **Pull requests: Write** permission for creating PRs
- **Issues: Write** permission for commenting on issues

## Usage

### Automatic Integration (Recommended)

The main Studio pipeline automatically handles prompt generation and PR creation:

```bash
# Run Studio with all integrations enabled
go run cmd/studio/main.go
```

When personas are updated, the system will:
1. Detect the update during the next pipeline run
2. Generate prompts automatically
3. Create PR with all generated content
4. Comment on the original issue

### Manual PR Generation

Generate prompts and create PR for a specific persona:

```bash
# Generate all prompts with PR creation
go run cmd/prompt-generator-pr/main.go -persona "David Attenborough" -pr

# Generate specific prompt types with PR
go run cmd/prompt-generator-pr/main.go -persona "David Attenborough" -type platform -pr

# Link to a specific issue
go run cmd/prompt-generator-pr/main.go -persona "David Attenborough" -issue 123 -pr

# Force regeneration with PR
go run cmd/prompt-generator-pr/main.go -persona "David Attenborough" -force -pr
```

### README Trigger Method

Add trigger markers to persona READMEs to automatically generate PRs:

```html
<!-- To generate all prompts with PR -->
<!-- GENERATE:prompts -->

<!-- To generate platform prompts with PR -->
<!-- GENERATE:platform_prompts -->

<!-- To generate variation prompts with PR -->
<!-- GENERATE:variation_prompts -->
```

The system will detect these markers and create PRs automatically.

### Monitoring Mode

Run continuous monitoring for automatic PR creation:

```bash
# Monitor every 5 minutes and create PRs as needed
go run cmd/prompt-generator-pr/main.go -interval 5m

# Run once for all pending prompts
go run cmd/prompt-generator-pr/main.go -once -pr
```

## PR Workflow

### 1. Automatic PR Creation

```
[User updates synthesized.md or adds trigger]
         ↓
[Studio detects change in next pipeline run]
         ↓
[Gemini generates 7 optimized prompts]
         ↓
[System creates branch and commits files]
         ↓
[PR created with comprehensive description]
         ↓
[Comment added to original issue (if applicable)]
```

### 2. Review Process

1. **Automated PR Creation**: Studio creates the PR with all generated content
2. **Review Generated Prompts**: Team reviews the prompt quality and optimization
3. **Test Prompts**: Optional testing of prompts with actual AI platforms
4. **Merge PR**: Merge the PR to make prompts available
5. **Automatic Cleanup**: GitHub automatically cleans up the feature branch

### 3. Issue Integration

When linked to an issue:
- **Issue Comment**: Automatic comment with PR link and summary
- **Issue Closing**: PR description includes "Closes #[issue_number]"
- **Linking**: Clear connection between issue request and generated prompts

## Generated PR Example

**Title**: `Update prompts for persona: David Attenborough`

**Body**:
```markdown
This PR updates the generated prompts for **David Attenborough** with the latest content from the synthesized persona.

## 🎯 Generated Prompts

Successfully generated **7 prompts** optimized for different platforms and use cases:

### Platform-Specific Prompts
- ✅ **ChatGPT System Prompt**
- ✅ **Claude System Prompt** 
- ✅ **Gemini System Prompt**
- ✅ **Discord Bot Personality**
- ✅ **Character.AI Character**

### Prompt Variations
- ✅ **Condensed Prompt-Ready Version**
- ✅ **Alternative Variations**

## 📁 Files Updated

- `prompts/` - All generated prompt files
- `README.md` - Updated with prompt links and usage instructions
- `.assets_status.json` - Updated asset generation tracking

## 🚀 How to Use

1. **Browse the prompts** in the `prompts/` directory
2. **Copy the appropriate prompt** for your AI platform
3. **Paste into your AI interface** (ChatGPT, Claude, etc.)
4. **Start conversing** with the persona

## ⚙️ Generation Details

- **Model:** Gemini 2.0 Flash
- **Generated:** 2025-07-13 10:30:45 UTC
- **Source:** synthesized.md
- **Templates:** Platform-specific optimization

## 🔗 Related Issue

Closes #123

---
*This PR was created automatically by Studio - Multi-AI Persona Generation Pipeline*
```

## Features

### Intelligent Branch Naming
- **Pattern**: `prompts/{persona-name}-{timestamp}`
- **Example**: `prompts/david-attenborough-20250713-103045`
- **Collision Prevention**: Timestamp ensures unique branch names

### Comprehensive File Management
- **Prompt Files**: All 7 generated prompts with metadata headers
- **README Updates**: Automatic links and usage instructions
- **Status Tracking**: Updated asset generation status
- **Conflict Resolution**: Handles existing files gracefully

### Quality Assurance
- **Validation**: Each prompt validated before inclusion
- **Error Handling**: Failed generations documented and skipped
- **Metadata**: Rich metadata headers with generation details
- **Consistency**: Standardized formatting across all prompts

### Team Collaboration
- **Clear Descriptions**: Detailed PR descriptions with usage instructions
- **File Organization**: Logical organization of generated content
- **Review Process**: Easy review of generated prompts
- **Issue Integration**: Clear connection to original requests

## Troubleshooting

### Common Issues

**PR Creation Fails**
- Verify GitHub token has correct permissions
- Check repository settings and access rights
- Ensure branch doesn't already exist

**Prompts Not Generated**
- Verify Gemini API key is valid and has quota
- Check synthesized.md file exists and is readable
- Review asset status for error messages

**Missing Files in PR**
- Check file validation - invalid prompts are skipped
- Review generation logs for errors
- Verify all required directories exist

**Permission Errors**
- Ensure GitHub token has repository write access
- Check organization permissions if using org repos
- Verify personas repository allows external PRs

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
go run cmd/prompt-generator-pr/main.go -persona "David Attenborough" -pr -log debug
```

This provides:
- Prompt generation details
- GitHub API request/response information
- File operation logging
- Branch creation and PR submission details

## Best Practices

### For Users
1. **Review Before Merge**: Always review generated prompts for quality
2. **Test Prompts**: Test prompts with actual AI platforms when possible
3. **Update Triggers**: Use README triggers for hands-off automation
4. **Monitor Status**: Check asset status for generation tracking

### For Developers
1. **Error Handling**: Implement graceful error handling for API failures
2. **Rate Limiting**: Respect GitHub and Gemini API rate limits
3. **Validation**: Always validate generated content before submission
4. **Logging**: Comprehensive logging for debugging and monitoring

### For Teams
1. **Review Process**: Establish clear review process for generated PRs
2. **Quality Standards**: Define quality standards for prompt acceptance
3. **Testing**: Implement testing procedures for generated prompts
4. **Documentation**: Keep documentation updated with process changes

## Integration Points

### Studio Pipeline
- **Automatic Detection**: Integrated with main pipeline for automatic processing
- **Asset Monitoring**: Uses existing asset system for trigger detection
- **State Management**: Leverages existing state tracking infrastructure

### Existing Workflows
- **Issue Processing**: Integrates with existing issue-based persona creation
- **PR Comments**: Compatible with existing PR comment feedback system
- **Status Tracking**: Uses existing asset status tracking system

### External Tools
- **GitHub Actions**: Compatible with GitHub Actions workflows
- **Webhooks**: Can be triggered by webhook events
- **API Integration**: Programmatic access through Studio's API endpoints

## Future Enhancements

### Planned Features
- **Batch PR Creation**: Multiple personas in single PR
- **Template Customization**: Custom PR templates per team
- **A/B Testing**: Multiple prompt variants in single PR
- **Quality Scoring**: Automatic quality assessment of generated prompts

### Integration Opportunities
- **CI/CD Integration**: Automatic testing of generated prompts
- **Review Automation**: Automated prompt quality checks
- **Deployment Integration**: Automatic deployment of approved prompts
- **Analytics Integration**: Usage tracking for generated prompts