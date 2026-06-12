# User-Supplied Personas

## Overview

Studio now supports including user-supplied personas in the synthesis process. This allows users to contribute their own version of a persona, which will be combined with the AI-generated versions to create an even more comprehensive result.

## How It Works

1. **Input**: Users can optionally include their own persona between `[[[` and `]]]` markers in the GitHub issue
2. **Processing**: The user persona is extracted and passed to the synthesis step
3. **Synthesis**: Gemini combines the user persona with the 4 AI-generated versions (Claude, Gemini, Grok, GPT-4)
4. **Storage**: The user persona is saved as `user_supplied.md` in the raw outputs folder

## Usage

### Basic Format

```markdown
Title: Create Persona: [Name]

[[[
Your complete persona content here...
]]]
```

### With Additional Context

```markdown
Title: Create Persona: [Name]

<<<
**Background & Context:**
Additional context for AI providers...

**Personality Traits:**
Specific traits to emphasize...
>>>

[[[
Your complete persona goes here...
]]]
```

## Benefits

1. **Customization**: Users can provide their specific interpretation or requirements
2. **Refinement**: Existing personas can be enhanced with user knowledge
3. **Domain Expertise**: Users with specialized knowledge can contribute
4. **Iterative Improvement**: Users can refine personas based on previous outputs
5. **Hybrid Approach**: Combines human insight with AI capabilities

## Example Use Cases

### 1. Domain Expert Contribution
A historian provides their detailed persona of a historical figure based on primary sources.

### 2. Corporate Persona
A company provides their official brand persona to ensure consistency.

### 3. Fictional Character
An author provides the canonical version of their character.

### 4. Persona Refinement
After seeing AI outputs, a user provides an improved version incorporating the best elements.

## Synthesis Process

When a user persona is provided:

1. All 4 AI providers generate their versions based on the issue content
2. The user persona is included as a 5th input to the synthesis
3. Gemini combines all 5 versions with equal consideration
4. The synthesis prompt indicates that a user-supplied version is included

## File Structure

```
personas/[name]/
├── raw/
│   ├── claude.md
│   ├── gemini.md
│   ├── grok.md
│   ├── gpt.md
│   └── user_supplied.md  ← User's persona stored here
├── synthesized.md        ← Combines all 5 inputs
```

## Important Notes

- The user persona is optional - issues work normally without it
- The placeholder text `[Paste your complete persona here]` is ignored
- Empty user personas are skipped
- User personas are treated with equal weight during synthesis
- The PR description indicates when a user persona was included

## Future Enhancements

- Weighted synthesis (giving user personas more influence)
- Multiple user personas from different contributors
- User persona validation and formatting
- Feedback loop for user persona improvement