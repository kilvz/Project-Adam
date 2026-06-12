# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Studio is a multi-AI persona generation pipeline that monitors GitHub issues in `twin2ai/personas` and automatically generates comprehensive user personas using four AI providers (Claude 4 Opus, Gemini 2.5 Pro, Grok 2, GPT-4 Turbo) running in parallel. The system uses Gemini to intelligently combine all responses into a superior final persona.

## Build and Development Commands

```bash
# Build the application
go build -o studio cmd/studio/main.go

# Run in development
go run cmd/studio/main.go

# Run tests  
go test ./...

# Format code
go fmt ./...

# Lint (if golangci-lint available)
golangci-lint run

# Cross-compile for Linux
GOOS=linux GOARCH=amd64 go build -o studio-linux cmd/studio/main.go

# Docker deployment
docker-compose up -d
docker-compose logs -f studio

# Batch persona generation
./studio batch names.txt                # Process personas from file
./studio batch -force names.txt         # Force regeneration even if exists
```

## Architecture Overview

### Core Pipeline Flow
1. **Issue Monitoring**: `internal/pipeline` orchestrates polling `twin2ai/personas` for issues labeled `create-persona` or `update-persona`
2. **Multi-Provider Generation**: `internal/multiprovider` sends identical prompts to all four AI providers simultaneously (create only)
3. **Individual Storage**: Each AI response is stored in `artifacts/{provider}/` for analysis
4. **AI-Powered Synthesis**: Gemini combines all responses using prompts from `prompts/persona_combination.txt`
5. **PR Creation**: Generated personas are submitted as pull requests to `twin2ai/personas`
6. **Comment Processing**: Feedback comments trigger regeneration using keywords like "regenerate", "truncated"
7. **Update Processing**: User-submitted updates are synthesized with existing personas (no multi-provider generation)
8. **Batch Processing**: Process multiple personas from a text file via `studio batch` command

### Key Components
- **`internal/pipeline/pipeline.go`**: Main orchestration, handles both new issues and PR comment feedback
- **`internal/pipeline/update_persona.go`**: Handles persona update requests
- **`internal/pipeline/batch_pipeline.go`**: Batch processing of personas from text files
- **`internal/multiprovider/generator.go`**: Parallel AI provider coordination and response combination
- **`internal/multiprovider/enhanced_generator.go`**: Structured persona generation and updates
- **`internal/{claude,gemini,grok,gpt}/client.go`**: Individual AI provider clients with model-specific configurations
- **`internal/github/client.go`**: GitHub API integration for issue/PR management
- **`internal/github/update_persona.go`**: Persona update PR creation
- **`internal/config/config.go`**: Environment-based configuration management

### AI Model Configuration
- **Claude 4 Opus**: `claude-opus-4-20250514` with 20,000 max tokens
- **Gemini 2.5 Pro**: `gemini-2.5-pro` with 20,000 max tokens via `generationConfig`
  - Temperature: 0.7 for persona generation, 0.3 for synthesis tasks
- **Grok 2**: `grok-2-1212` with 20,000 max tokens
- **GPT-4 Turbo**: `gpt-4-turbo` with 4,000 max tokens (context limit considerations)

### Data Flow
- **Input**: GitHub issues with `create-persona` label
- **Processing**: Parallel AI generation → Individual artifact storage → Gemini synthesis
- **Synthesis Chain**: Raw personas → Synthesized persona
- **Output**: Combined persona in PR + individual responses in `artifacts/`
- **Feedback Loop**: PR comments trigger regeneration with enhanced prompts

### Prompt Templates
- **Location**: `prompts/` directory contains all generation prompts
- **Customizable**: Edit prompt files without modifying code
- **Templates**: `persona_combination.txt`, `persona_combination_feedback.txt`

### State Management
- **Processed Issues**: Tracked in `data/processed_issues.txt` to prevent duplicate processing
- **Processed Comments**: Tracked in `data/processed_comments.txt` using `{PR#}-{CommentID}` format
- **Batch Processed Names**: Tracked in `data/batch_processed_names.txt` to avoid reprocessing
- **Artifacts**: Timestamped storage in `artifacts/{provider}/` with separate dirs for feedback iterations

### Configuration Requirements
Environment variables in `.env`:
- GitHub token with repo permissions
- API keys for all four AI providers  
- Model names for each provider
- Poll interval and logging configuration

### Error Handling Patterns
- **Fault Tolerance**: Pipeline continues if individual AI providers fail
- **Timeout Management**: 600s (10 minutes) timeout for all AI providers
- **Context Length**: GPT-4 uses shorter prompts due to 8K context limit
- **Retry Logic**: Comments marked as processed immediately to prevent duplicate regeneration

### Logging Strategy
- **Dual Output**: Console (colored text) + file (`logs/studio.log`)
- **Debug Mode**: Set `LOG_LEVEL=debug` for detailed API request/response logging
- **Provider-Specific**: Each AI client has comprehensive request/response logging for debugging