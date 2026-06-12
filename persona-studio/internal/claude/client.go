package claude

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

const anthropicAPIURL = "https://api.anthropic.com/v1/messages"

type Client struct {
	apiKey     string
	model      string
	httpClient *http.Client
	logger     *logrus.Logger
	promptPath string
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Request struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature float64   `json:"temperature,omitempty"`
}

type Response struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
}

func NewClient(apiKey, model string, logger *logrus.Logger) *Client {
	return &Client{
		apiKey: apiKey,
		model:  model,
		httpClient: &http.Client{
			Timeout: 600 * time.Second, // 10 minutes, matching other providers
		},
		logger:     logger,
		promptPath: "prompts/persona_generation.txt",
	}
}

func (c *Client) loadPromptTemplate() (string, error) {
	data, err := os.ReadFile(c.promptPath)
	if err != nil {
		// If file doesn't exist, return a default prompt
		if os.IsNotExist(err) {
			c.logger.Warn("Prompt file not found, using default prompt")
			return c.getDefaultPrompt(), nil
		}
		return "", fmt.Errorf("failed to read prompt file: %w", err)
	}

	prompt := strings.TrimSpace(string(data))
	if prompt == "" {
		return c.getDefaultPrompt(), nil
	}

	return prompt, nil
}

func (c *Client) getDefaultPrompt() string {
	return `Based on the following GitHub issue, create a detailed user persona:

{{ISSUE_CONTENT}}

Please create a comprehensive persona that includes:
1. Name and basic demographics
2. Background and context
3. Goals and motivations
4. Pain points and challenges
5. Technical proficiency level
6. Preferred tools and platforms
7. Behavioral patterns
8. Success criteria

Format the output as a well-structured Markdown document suitable for documentation.

{{TEMPLATE}}`
}

func (c *Client) GeneratePersona(ctx context.Context, issueContent string, template string) (string, error) {
	c.logger.Info("Starting Claude persona generation")

	// Load prompt template
	promptTemplate, err := c.loadPromptTemplate()
	if err != nil {
		c.logger.Warnf("Failed to load prompt template: %v", err)
		return "", fmt.Errorf("failed to load prompt template: %w", err)
	}
	c.logger.Debug("Loaded prompt template successfully")

	// Replace placeholders in the prompt
	prompt := strings.ReplaceAll(promptTemplate, "{{ISSUE_CONTENT}}", issueContent)

	if template != "" {
		templateSection := fmt.Sprintf("\n\nUse this template as a guide:\n%s", template)
		prompt = strings.ReplaceAll(prompt, "{{TEMPLATE}}", templateSection)
		c.logger.Debug("Added template to prompt")
	} else {
		prompt = strings.ReplaceAll(prompt, "{{TEMPLATE}}", "")
		c.logger.Debug("No template provided")
	}

	c.logger.Infof("Using Claude model: %s", c.model)
	c.logger.Debugf("Prompt length: %d characters", len(prompt))

	// Check if prompt is too large
	if len(prompt) > 50000 {
		c.logger.Warnf("Large prompt detected (%d chars), this might cause API issues", len(prompt))
	}

	request := Request{
		Model: c.model,
		Messages: []Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens:   20000,
		Temperature: 0.7,
	}

	body, err := json.Marshal(request)
	if err != nil {
		c.logger.Errorf("Failed to marshal request: %v", err)
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}
	c.logger.Debugf("Request body length: %d bytes", len(body))

	req, err := http.NewRequestWithContext(ctx, "POST", anthropicAPIURL, bytes.NewReader(body))
	if err != nil {
		c.logger.Errorf("Failed to create HTTP request: %v", err)
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	c.logger.Info("Sending request to Claude API...")
	c.logger.Debugf("Request URL: %s", anthropicAPIURL)
	c.logger.Debugf("Request headers: Content-Type=%s, x-api-key=%s..., anthropic-version=%s",
		req.Header.Get("Content-Type"),
		c.apiKey[:min(10, len(c.apiKey))]+"...",
		req.Header.Get("anthropic-version"))

	// Add timeout context for the request
	requestStart := time.Now()
	resp, err := c.httpClient.Do(req)
	requestDuration := time.Since(requestStart)

	c.logger.Infof("HTTP request completed in %v", requestDuration)

	if err != nil {
		c.logger.Errorf("HTTP request failed after %v: %v", requestDuration, err)
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	c.logger.Infof("Received response with status: %d", resp.StatusCode)
	c.logger.Debugf("Response headers: %+v", resp.Header)

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.logger.Errorf("Failed to read response body: %v", err)
		return "", fmt.Errorf("failed to read response body: %w", err)
	}
	c.logger.Debugf("Response body length: %d bytes", len(responseBody))

	if resp.StatusCode != http.StatusOK {
		c.logger.Errorf("API request failed with status %d: %s", resp.StatusCode, string(responseBody))
		return "", fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(responseBody))
	}

	c.logger.Debug("Parsing Claude API response...")
	var response Response
	if err := json.Unmarshal(responseBody, &response); err != nil {
		c.logger.Errorf("Failed to decode response JSON: %v", err)
		c.logger.Debugf("Raw response body: %s", string(responseBody))
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	c.logger.Infof("Response contains %d content items", len(response.Content))

	if len(response.Content) == 0 {
		c.logger.Error("Empty content array in Claude response")
		c.logger.Debugf("Full response: %+v", response)
		return "", fmt.Errorf("empty response from Claude API")
	}

	// Log content types for debugging
	for i, content := range response.Content {
		c.logger.Debugf("Content[%d]: type=%s, text_length=%d", i, content.Type, len(content.Text))
	}

	// With extended thinking, we want the final text content, not the thinking
	for _, content := range response.Content {
		if content.Type == "text" {
			c.logger.Infof("Found text content, length: %d characters", len(content.Text))
			if content.Text == "" {
				c.logger.Warn("Text content is empty")
			}
			return content.Text, nil
		}
	}

	// Fallback to first content if no specific text type found
	c.logger.Warn("No 'text' type content found, using first content item")
	firstContent := response.Content[0].Text
	c.logger.Infof("Using fallback content, length: %d characters", len(firstContent))
	if firstContent == "" {
		c.logger.Error("Fallback content is also empty")
	}
	return firstContent, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
