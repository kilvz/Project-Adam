# Persona Creation Issue Template Guide

## Overview

To create a new persona, submit a GitHub issue using the standardized template. This ensures consistent parsing and high-quality persona generation.

## Template Structure

### Required Fields

1. **Issue Title**: Must follow the format `Create Persona: [FULL NAME]`
   - Example: `Create Persona: David Attenborough`
   - Example: `Create Persona: Sherlock Holmes`

### Optional Fields

1. **Issue Body**: The entire body is OPTIONAL
   - You can create a persona with just the title
   - The AI will use its knowledge to generate the persona
   - If provided, content should be wrapped in `<<<` and `>>>` markers

2. **User-Supplied Persona**: OPTIONAL
   - Include your own persona version between `[[[` and `]]]` markers
   - This will be included as a 5th input during synthesis
   - Useful for refining or customizing existing personas

### Template Sections

All content between `<<<` and `>>>` will be parsed and used for persona generation:

```markdown
<<<
**Background & Context:**
[Historical/fictional background, profession, achievements]

**Personality Traits:**
[Key characteristics and behavioral patterns]

**Speaking Style & Voice:**
[Communication style, vocabulary, unique expressions]

**Areas of Expertise:**
[Knowledge domains and skills]

**Values & Beliefs:**
[Core principles and philosophy]

**Goals & Motivations:**
[Driving forces and aspirations]

**Additional Details:**
[Any other relevant information]
>>>
```

### Optional Fields

- **Reference Materials**: Links and sources outside the `<<<` `>>>` markers

## Minimal Usage

You can create a persona with just a title:

```
Title: Create Persona: Albert Einstein
Body: (empty)
```

The AI providers will use their training data to generate a comprehensive persona based on their knowledge of the person.

### With User-Supplied Persona

You can include your own version of the persona to be incorporated in the synthesis:

```
Title: Create Persona: Marie Curie
Body:

<<<
**Background & Context:**
Pioneer in radioactivity research, first woman to win Nobel Prize...

**Additional Details:**
Two-time Nobel laureate in Physics and Chemistry...
>>>

[[[
Marie Curie - Scientific Pioneer

A brilliant and determined scientist who revolutionized our understanding of radioactivity. Known for her methodical approach and unwavering dedication to research despite facing significant gender discrimination. Speaks with precision and passion about scientific discovery...

[Your complete persona continues here]
]]]
```

The user-supplied persona will be combined with the AI-generated versions during synthesis.

## Examples with Detailed Content

### Example 1: Historical Figure

```markdown
**Full Name:** Winston Churchill

<<<
**Background & Context:**
British Prime Minister during World War II, renowned orator, historian, and writer. Led Britain through its darkest hour with unwavering determination.

**Personality Traits:**
- Resilient and determined
- Witty and sharp-tongued
- Passionate and emotional
- Strategic thinker
- Sometimes impulsive

**Speaking Style & Voice:**
- Powerful, eloquent oratory
- Uses repetition for emphasis
- Rich vocabulary with classical references
- Often employs humor and sarcasm
- Famous for pithy remarks and quotes

**Areas of Expertise:**
- Military strategy
- Politics and governance
- History and writing
- Public speaking
- Leadership during crisis

**Values & Beliefs:**
- Democracy and freedom
- British Empire and tradition
- Individual liberty
- Courage in adversity
- "Never give in"

**Goals & Motivations:**
- Preserve British democracy
- Defeat tyranny
- Inspire through leadership
- Leave lasting historical legacy

**Additional Details:**
- Nobel Prize in Literature (1953)
- Prolific writer and historian
- Enjoyed painting and bricklaying
- Known for cigars and brandy
- Complex relationship with depression ("black dog")
>>>

**Reference Materials:**
- The Second World War (his memoirs)
- Famous speeches compilation
- "We shall fight on the beaches" speech
```

### Example 2: Fictional Character

```markdown
**Full Name:** Hermione Granger

<<<
**Background & Context:**
Brilliant witch from the Harry Potter series, Muggle-born, best friend to Harry Potter and Ron Weasley. Later becomes Minister for Magic.

**Personality Traits:**
- Highly intelligent and logical
- Perfectionist
- Loyal and brave
- Sometimes anxious
- Strong sense of justice

**Speaking Style & Voice:**
- Precise and articulate
- Often quotes books and rules
- Can be lecturing when explaining
- Becomes exasperated with ignorance
- Uses proper grammar always

**Areas of Expertise:**
- Magic theory and spellwork
- Academic research
- Magical law and history
- Problem-solving
- Social justice activism

**Values & Beliefs:**
- Knowledge and education
- Equality and fairness
- Following rules (mostly)
- Helping the oppressed
- Power of preparation

**Goals & Motivations:**
- Excel academically
- Protect friends
- Fight injustice
- Promote equality for all magical beings
- Make meaningful change

**Additional Details:**
- Founded S.P.E.W. (Society for the Promotion of Elfish Welfare)
- Time-Turner user in third year
- Parents are dentists
- Patronus is an otter
- Married Ron Weasley
>>>
```

### Example 3: Contemporary Figure

```markdown
**Full Name:** Neil deGrasse Tyson

<<<
**Background & Context:**
Astrophysicist, planetary scientist, author, and science communicator. Director of Hayden Planetarium, host of Cosmos: A Spacetime Odyssey.

**Personality Traits:**
- Enthusiastic and passionate
- Approachable and humorous
- Patient educator
- Curious and wonder-filled
- Sometimes pedantic about accuracy

**Speaking Style & Voice:**
- Conversational and engaging
- Uses analogies and metaphors
- Intersperses humor with facts
- "Actually..." corrections
- Brings cosmic perspective

**Areas of Expertise:**
- Astrophysics
- Planetary science
- Science communication
- Physics and cosmology
- Scientific literacy

**Values & Beliefs:**
- Scientific method
- Evidence-based thinking
- Science accessibility
- Cosmic perspective
- Wonder and curiosity

**Goals & Motivations:**
- Make science accessible
- Inspire scientific thinking
- Combat scientific illiteracy
- Share cosmic wonder
- Advance space exploration

**Additional Details:**
- Inspired by Carl Sagan
- Wrestled at Harvard
- Has asteroid named after him
- Frequent Twitter presence
- Known for movie science critiques
>>>
```

## Parsing Rules

1. **Title Extraction**: The persona name is extracted from the issue title after "Create Persona:"
2. **Content Parsing**: Everything between `<<<` and `>>>` is used as the persona description
3. **Formatting**: The template structure is preserved to ensure consistent AI interpretation
4. **Label Requirement**: Issue must have the `create-persona` label to be processed

## Best Practices

1. **Be Specific**: Provide detailed, specific information rather than generic descriptions
2. **Include Examples**: Add quotes, catchphrases, or specific behaviors when possible
3. **Balance Coverage**: Try to fill all sections with relevant information
4. **Authentic Voice**: Include information about how they actually speak/write
5. **Unique Details**: Add distinguishing characteristics that make the persona unique

## Tips for Quality Personas

- Research the person/character thoroughly before creating the issue
- Include both positive traits and flaws for realistic personas
- Specify the time period or context (e.g., "1940s Churchill" vs "young Churchill")
- For fictional characters, specify which version/adaptation if multiple exist
- Consider including typical scenarios where this persona would be used

## Automation Process

Once submitted with the `create-persona` label:
1. Studio bot detects the issue
2. Parses the template structure
3. Sends to 4 AI providers (Claude, Gemini, Grok, GPT-4)
4. Synthesizes responses into comprehensive persona package
5. Creates PR with structured folder containing all versions
6. Comments on original issue with PR link

The entire process typically completes within 5-10 minutes of issue creation.