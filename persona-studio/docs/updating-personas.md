# Updating Existing Personas

## Overview

Studio now supports updating existing personas with user-provided content. This feature allows you to enhance or modify personas without regenerating from scratch.

## How It Works

1. **Submit Update**: Create a GitHub issue with your updated persona
2. **Synthesis**: Studio retrieves the existing persona and synthesizes it with your version
3. **Pull Request**: A PR is created with only the synthesized version
4. **Simplified Process**: Only the synthesized version is updated

## Creating an Update Request

### Issue Format

```markdown
Title: Update Persona: [Existing Persona Name]
Labels: update-persona

Body:
[[[
Your complete updated persona content here...
]]]
```

### Requirements

- **Title**: Must start with "Update Persona:" followed by the exact persona name
- **Label**: Must have `update-persona` label
- **Body**: Your updated persona (optionally wrapped in `[[[` `]]]` markers)

## Example

```markdown
Title: Update Persona: David Attenborough
Labels: update-persona

[[[
Sir David Attenborough - Natural World Narrator and Environmental Advocate

A legendary broadcaster and natural historian whose gentle, wonder-filled narration has brought the natural world into millions of homes. At 97, his dedication to documenting Earth's biodiversity and advocating for conservation remains unwavering.

## Voice & Communication Style
- Speaks in measured, thoughtful tones with perfect diction
- Uses poetic language to describe natural phenomena
- Whispers to avoid disturbing wildlife
- Employs dramatic pauses for effect
- British accent with refined pronunciation

[Rest of updated persona...]
]]]
```

## Synthesis Process

When you submit an update:

1. **Retrieval**: Studio fetches the existing persona from the repository
2. **Synthesis**: Gemini (temperature 0.3) intelligently merges both versions
3. **Conflict Resolution**: User-provided content takes precedence in conflicts
4. **Enhancement**: New information is incorporated while preserving valuable existing details

## What Gets Updated

✅ **Updated**:
- The main synthesized persona file (`synthesized.md` or main `.md` file)

❌ **Not Updated**:
- Raw AI provider outputs

## Benefits

1. **Iterative Improvement**: Refine personas based on usage experience
2. **User Control**: Direct input into the final persona
3. **Preservation**: Valuable existing content is retained
4. **Efficiency**: Faster than full regeneration
5. **Focused Updates**: Only synthesis is performed

## Best Practices

1. **Complete Persona**: Provide a complete updated version, not just changes
2. **Maintain Structure**: Keep consistent formatting with the original
3. **Add Context**: Include why changes are being made in your version
4. **Review Original**: Check the existing persona before updating

## Error Handling

If your update fails, check:
- Persona name matches exactly (case-sensitive)
- Persona exists in the repository
- Issue format is correct
- Content is provided in the body

## Limitations

- Only works with existing personas
- Cannot update raw AI outputs
- Platform adaptations must be regenerated separately if needed
- One persona per update issue

## Future Enhancements

- Batch updates for multiple personas
- Selective section updates
- Version history tracking
- Automatic platform adaptation updates