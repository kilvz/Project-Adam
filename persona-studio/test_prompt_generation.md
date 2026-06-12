# Test Prompt Generation System

## Quick Test Steps

### 1. Set up environment
```bash
export GOOGLE_API_KEY="your_gemini_api_key"
export GEMINI_MODEL="gemini-2.0-flash-exp"
```

### 2. Create a test persona
```bash
mkdir -p personas/test_persona
echo "# Test Persona

This is Albert Einstein, the renowned theoretical physicist known for his theory of relativity and contributions to quantum mechanics. He is curious, imaginative, and deeply philosophical about the nature of the universe." > personas/test_persona/synthesized.md
```

### 3. Generate prompts manually
```bash
# Generate all prompts for the test persona
./prompt-generator -persona "test_persona" -log debug

# Or generate only platform prompts
./prompt-generator -persona "test_persona" -type platform -log debug

# Or generate only variation prompts  
./prompt-generator -persona "test_persona" -type variation -log debug
```

### 4. Check generated output
```bash
ls -la personas/test_persona/prompts/
cat personas/test_persona/README.md
cat personas/test_persona/.assets_status.json
```

### 5. Test with README triggers
Add this to the persona's README.md:
```html
<!-- GENERATE:prompts -->
```

Then run the asset monitor:
```bash
./asset-monitor -once -log debug
```

## Expected Output Structure

After running the prompt generator, you should see:

```
personas/test_persona/
├── synthesized.md              # Original synthesized persona
├── README.md                   # Updated with prompt links
├── .assets_status.json         # Asset tracking status
└── prompts/                    # Generated prompts directory
    ├── chatgpt.md              # ChatGPT system prompt
    ├── claude.md               # Claude system prompt  
    ├── gemini.md               # Gemini system prompt
    ├── discord.md              # Discord bot personality
    ├── characterai.md          # Character.AI character
    ├── condensed.md            # Prompt-ready version
    └── alternative.md          # Alternative variations
```

## Generated Prompt Format

Each prompt file includes:
- Metadata header with generation timestamp
- Platform-optimized prompt content
- Footer with generation information

Example:
```markdown
# ChatGPT System Prompt

> **Generated:** 2025-07-13T10:30:00Z  
> **Persona:** test_persona  
> **Type:** chatgpt  
> **Source:** synthesized.md  

---

You are Albert Einstein, the renowned theoretical physicist...

---

*Generated automatically by Studio using Gemini 2.0 Flash*  
*Last updated: 2025-07-13 10:30:00 UTC*
```

## Integration Test

To test the full integration:

1. **Update synthesized.md** - Modify the synthesized persona content
2. **Trigger generation** - Either use README markers or run manually  
3. **Verify updates** - Check that prompts are regenerated and README is updated
4. **Check status** - Verify asset status tracking shows completion

## Troubleshooting

### Common issues:
- **API Key**: Ensure GOOGLE_API_KEY is set correctly
- **Permissions**: Check write permissions to persona directories
- **File paths**: Ensure synthesized.md exists in the persona folder
- **Rate limits**: Be aware of Gemini API rate limiting

### Debug mode:
```bash
./prompt-generator -persona "test_persona" -log debug
```

This provides detailed logging for:
- Template loading
- API requests/responses  
- File operations
- Validation steps