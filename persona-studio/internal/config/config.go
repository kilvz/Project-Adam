package config

import (
	"os"
	"time"
)

type Config struct {
	GitHub   GitHubConfig
	AI       AIConfig
	Pipeline PipelineConfig
}

type GitHubConfig struct {
	Token         string
	Owner         string
	Repo          string
	PersonasOwner string
	PersonasRepo  string
	PersonaLabel  string
}

type AIConfig struct {
	Claude ClaudeConfig
	Gemini GeminiConfig
	Grok   GrokConfig
	GPT    GPTConfig
}

type ClaudeConfig struct {
	APIKey string
	Model  string
}

type GeminiConfig struct {
	APIKey string
	Model  string
}

type GrokConfig struct {
	APIKey string
	Model  string
}

type GPTConfig struct {
	APIKey string
	Model  string
}

type PipelineConfig struct {
	PollInterval time.Duration
	DataDir      string
	LogDir       string
}

func Load() (*Config, error) {
	pollInterval, err := time.ParseDuration(getEnv("POLL_INTERVAL", "5m"))
	if err != nil {
		pollInterval = 5 * time.Minute
	}

	return &Config{
		GitHub: GitHubConfig{
			Token:         getEnv("GITHUB_TOKEN", ""),
			Owner:         getEnv("GITHUB_OWNER", ""),
			Repo:          getEnv("GITHUB_REPO", ""),
			PersonasOwner: getEnv("PERSONAS_OWNER", "twin2ai"),
			PersonasRepo:  getEnv("PERSONAS_REPO", "personas"),
			PersonaLabel:  getEnv("PERSONA_LABEL", "create-persona"),
		},
		AI: AIConfig{
			Claude: ClaudeConfig{
				APIKey: getEnv("ANTHROPIC_API_KEY", ""),
				Model:  getEnv("CLAUDE_MODEL", "claude-opus-4-20250514"),
			},
			Gemini: GeminiConfig{
				APIKey: getEnv("GOOGLE_API_KEY", ""),
				Model:  getEnv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
			},
			Grok: GrokConfig{
				APIKey: getEnv("GROK_API_KEY", ""),
				Model:  getEnv("GROK_MODEL", "grok-2-1212"),
			},
			GPT: GPTConfig{
				APIKey: getEnv("OPENAI_API_KEY", ""),
				Model:  getEnv("GPT_MODEL", "gpt-4"),
			},
		},
		Pipeline: PipelineConfig{
			PollInterval: pollInterval,
			DataDir:      getEnv("DATA_DIR", "./data"),
			LogDir:       getEnv("LOG_DIR", "./logs"),
		},
	}, nil
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
