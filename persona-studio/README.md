# Studio - AI-Powered Persona Pipeline in Go

Studio is a lightweight Go application that monitors GitHub issues in the `twin2ai/personas` repository and automatically generates user personas using Claude AI. It creates a seamless pipeline from issue creation to persona documentation in the same repository.

## Features

- **GitHub Integration**: Monitors issues in the `twin2ai/personas` repository for persona creation requests
- **Multi-AI Provider Support**: Generates personas using Claude, Gemini 2.0 Flash, Grok 2, and GPT-4 in parallel
- **AI-Powered Combination**: Uses Gemini to intelligently combine all four AI responses into a superior final persona
- **Artifact Storage**: Stores individual AI responses and combined results for analysis and comparison
- **Comment-Driven Regeneration**: Automatically regenerates personas based on feedback comments
- **Automated Workflow**: Creates personas in the same repository where issues are submitted
- **Single Binary**: Compiles to a single executable with no external dependencies
- **Docker Support**: Containerized deployment with Docker Compose
- **Configurable**: Flexible configuration via environment variables
- **Comprehensive Logging**: Detailed logging with configurable levels and dual output (console + file)

## Quick Start

### Prerequisites

- Go 1.21+ installed
- GitHub Personal Access Token with repo permissions
- AI Provider API Keys:
  - **Anthropic API Key** (Claude)
  - **Google API Key** (Gemini)
  - **Grok API Key** (Grok)
  - **OpenAI API Key** (GPT-4)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/twin2ai/studio.git
   cd studio
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

3. **Install dependencies**
   ```bash
   go mod download
   ```

4. **Run Studio**
   ```bash
   # Development
   go run cmd/studio/main.go
   
   # Or build and run
   go build -o studio cmd/studio/main.go
   ./studio
   ```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f studio

# Stop
docker-compose down
```

## Configuration

Configure Studio using environment variables in `.env`:

```env
# GitHub Configuration - Monitor twin2ai/personas
GITHUB_TOKEN=your_github_token
GITHUB_OWNER=twin2ai
GITHUB_REPO=personas

# AI Provider Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
GROK_API_KEY=your_grok_api_key
OPENAI_API_KEY=your_openai_api_key

# AI Model Configuration
CLAUDE_MODEL=claude-opus-4-20250514
GEMINI_MODEL=gemini-2.0-flash-exp
GROK_MODEL=grok-2-1212
GPT_MODEL=gpt-4

# Pipeline Configuration
POLL_INTERVAL=5m
PERSONA_LABEL=create-persona
LOG_LEVEL=info
```

## Usage

1. **Create an Issue**: In the `twin2ai/personas` repository, create an issue with:
   - The `create-persona` label
   - A descriptive title
   - Details about the persona you want to create

2. **Studio Processing**: Studio will:
   - Detect the new issue in `twin2ai/personas`
   - Generate personas using **all four AI providers in parallel**:
     - **Claude 4 Opus** with Extended Thinking
     - Gemini 2.0 Flash 
     - Grok 2
     - GPT-4
   - Store individual AI responses in `artifacts/` folder
   - Use **Gemini to intelligently combine** all responses into the best possible persona
   - Create a branch in the `twin2ai/personas` repository
   - Submit a pull request with the combined persona
   - Comment on the original issue with the PR link

3. **Comment-Driven Feedback**: Add comments with trigger words like:
   - "regenerate" - Triggers complete regeneration
   - "truncated" - Indicates output was cut off  
   - "needs more detail" - Requests expansion
   - "improve" - General improvement request
   
4. **Review and Merge**: Review the generated persona and merge the PR

## Multi-Provider Workflow

Studio's revolutionary approach combines four leading AI models:

### Parallel Generation Process
1. **Simultaneous Requests**: All four AI providers receive the same prompt simultaneously
2. **Individual Storage**: Each response is stored in `artifacts/{provider}/` for analysis
3. **Quality Diversity**: Different AI models excel at different aspects:
   - **Claude 4 Opus + Extended Thinking**: Deep reasoning, structured analysis with internal deliberation
   - **Gemini**: Strong at creative insights and synthesis
   - **Grok**: Unique perspectives and conversational tone
   - **GPT-4**: Comprehensive knowledge and balanced output

### AI-Powered Combination
4. **Intelligent Synthesis**: Gemini analyzes all four responses and creates an optimal combination
5. **Best of All Worlds**: The final persona incorporates the strongest elements from each provider
6. **Artifact Preservation**: All individual responses are preserved for comparison and analysis

### Benefits
- **Higher Quality**: Combined personas are more comprehensive than any single AI output
- **Reduced Bias**: Multiple AI perspectives reduce individual model limitations  
- **Fault Tolerance**: If one provider fails, others continue working
- **Transparency**: All individual responses are stored for review
- **Continuous Improvement**: Artifacts enable analysis of provider strengths

## Project Structure

```
studio/
├── cmd/studio/           # Application entry point
├── internal/
│   ├── config/          # Configuration management
│   ├── github/          # GitHub client
│   ├── claude/          # Claude API client
│   ├── gemini/          # Gemini API client
│   ├── grok/            # Grok API client
│   ├── gpt/             # GPT API client
│   ├── multiprovider/   # Multi-provider generation logic
│   ├── persona/         # Single-provider generation logic
│   └── pipeline/        # Main pipeline orchestration
├── pkg/models/          # Data models
├── templates/           # Persona templates
├── prompts/            # AI prompts
├── artifacts/          # AI provider responses
│   ├── claude/         # Claude responses
│   ├── gemini/         # Gemini responses
│   ├── grok/           # Grok responses
│   ├── gpt/            # GPT responses
│   └── combined/       # Final combined personas
├── data/               # Runtime data
├── logs/               # Application logs
├── docker-compose.yml  # Docker configuration
├── Dockerfile         # Container image
└── .env.example       # Environment template
```

## Development

### Running Tests

```bash
go test ./...
```

### Building for Production

```bash
# Build binary
go build -o studio cmd/studio/main.go

# Cross-compile for Linux
GOOS=linux GOARCH=amd64 go build -o studio-linux cmd/studio/main.go
```

### Code Quality

```bash
# Format code
go fmt ./...

# Lint (requires golangci-lint)
golangci-lint run
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request