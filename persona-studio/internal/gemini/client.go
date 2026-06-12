package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

type Client struct {
	apiKey     string
	model      string
	httpClient *http.Client
	logger     *logrus.Logger
}

type Content struct {
	Parts []Part `json:"parts"`
	Role  string `json:"role,omitempty"`
}

type Part struct {
	Text string `json:"text"`
}

type GenerationConfig struct {
	MaxOutputTokens int     `json:"maxOutputTokens"`
	Seed            int     `json:"seed,omitempty"`
	Temperature     float64 `json:"temperature,omitempty"`
}

type SafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

type Request struct {
	Contents         []Content        `json:"contents"`
	GenerationConfig GenerationConfig `json:"generationConfig"`
	SafetySettings   []SafetySetting  `json:"safetySettings,omitempty"`
}

type Response struct {
	Candidates []struct {
		Content       Content `json:"content"`
		FinishReason  string  `json:"finishReason,omitempty"`
		Index         int     `json:"index,omitempty"`
		SafetyRatings []struct {
			Category    string `json:"category"`
			Probability string `json:"probability"`
		} `json:"safetyRatings,omitempty"`
	} `json:"candidates"`
	PromptFeedback struct {
		SafetyRatings []struct {
			Category    string `json:"category"`
			Probability string `json:"probability"`
		} `json:"safetyRatings,omitempty"`
		BlockReason string `json:"blockReason,omitempty"`
	} `json:"promptFeedback,omitempty"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount,omitempty"`
		CandidatesTokenCount int `json:"candidatesTokenCount,omitempty"`
		TotalTokenCount      int `json:"totalTokenCount,omitempty"`
	} `json:"usageMetadata,omitempty"`
}

func NewClient(apiKey, model string, logger *logrus.Logger) *Client {
	return &Client{
		apiKey: apiKey,
		model:  model,
		httpClient: &http.Client{
			Timeout: 600 * time.Second,
		},
		logger: logger,
	}
}

// GeneratePersona generates a persona with standard temperature (0.7) for creative output
// This method is used for initial persona generation where creativity is desired
func (c *Client) GeneratePersona(ctx context.Context, prompt string) (string, error) {
	// Use standard temperature for creative persona generation
	return c.generateWithTemperature(ctx, prompt, 0.7)
}

// GenerateSynthesis generates content with lower temperature (0.3) for consistent synthesis
// Use this for combining multiple inputs, regenerating with feedback, or any task
// requiring predictable, faithful output rather than creative interpretation
func (c *Client) GenerateSynthesis(ctx context.Context, prompt string) (string, error) {
	// Use lower temperature for synthesis to get more consistent, predictable output
	return c.generateWithTemperature(ctx, prompt, 0.3)
}

// GeneratePersonaSynthesis is deprecated - use GenerateSynthesis instead
// DEPRECATED: This method will be removed in a future version
func (c *Client) GeneratePersonaSynthesis(ctx context.Context, prompt string) (string, error) {
	return c.GenerateSynthesis(ctx, prompt)
}

func (c *Client) generateWithTemperature(ctx context.Context, prompt string, temperature float64) (string, error) {
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", c.model, c.apiKey)

	request := Request{
		Contents: []Content{
			{
				Parts: []Part{
					{Text: prompt},
				},
			},
		},
		GenerationConfig: GenerationConfig{
			MaxOutputTokens: 20000,
			Temperature:     temperature,
			Seed:            12,
		},
	}

	c.logger.Debugf("Gemini request with temperature %.1f for prompt length: %d", temperature, len(prompt))

	body, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response Response
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if len(response.Candidates) == 0 || len(response.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("empty response from Gemini API")
	}

	return response.Candidates[0].Content.Parts[0].Text, nil
}

// GeneratePersonaPrompt generates a persona prompt using Gemini Flash with optimized settings
func (c *Client) GeneratePersonaPrompt(ctx context.Context, promptInput string) (string, error) {
	c.logger.Debugf("Generating persona prompt with Gemini Flash")

	request := Request{
		Contents: []Content{
			{
				Parts: []Part{
					{Text: promptInput},
				},
			},
		},
		GenerationConfig: GenerationConfig{
			Temperature:     0.3,   // Lower temperature for more consistent prompt generation
			MaxOutputTokens: 20000, // Increased to handle internal reasoning + output
		},
		SafetySettings: []SafetySetting{
			{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_ONLY_HIGH"},
			{Category: "HARM_CATEGORY_HATE_SPEECH", Threshold: "BLOCK_ONLY_HIGH"},
			{Category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", Threshold: "BLOCK_ONLY_HIGH"},
			{Category: "HARM_CATEGORY_DANGEROUS_CONTENT", Threshold: "BLOCK_ONLY_HIGH"},
		},
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// Use the working model and endpoint
	flashModel := "gemini-2.5-flash"
	endpoint := "v1"
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/%s/models/%s:generateContent?key=%s", endpoint, flashModel, c.apiKey)

	return c.tryGenerateWithModel(ctx, url, requestBody, flashModel)
}

// tryGenerateWithModel attempts to generate content with a specific model
func (c *Client) tryGenerateWithModel(ctx context.Context, url string, requestBody []byte, model string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(requestBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	c.logger.Debugf("Sending prompt generation request to %s", model)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Read the full response body for debugging
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	c.logger.Debugf("%s response status: %d, body length: %d bytes", model, resp.StatusCode, len(responseBody))

	if resp.StatusCode != http.StatusOK {
		c.logger.Errorf("%s API error response: %s", model, string(responseBody))
		return "", fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(responseBody))
	}

	// Log a snippet of the response for debugging (first 500 chars)
	if len(responseBody) > 500 {
		c.logger.Debugf("Response snippet: %s...", string(responseBody[:500]))
	} else {
		c.logger.Debugf("Full response: %s", string(responseBody))
	}

	var response Response
	if err := json.Unmarshal(responseBody, &response); err != nil {
		c.logger.Errorf("Failed to decode JSON response: %v", err)
		c.logger.Debugf("Raw response: %s", string(responseBody))
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	c.logger.Debugf("Response has %d candidates", len(response.Candidates))

	// Log usage metadata if available
	if response.UsageMetadata.TotalTokenCount > 0 {
		c.logger.Debugf("Token usage - Prompt: %d, Candidates: %d, Total: %d",
			response.UsageMetadata.PromptTokenCount,
			response.UsageMetadata.CandidatesTokenCount,
			response.UsageMetadata.TotalTokenCount)

		// Check for thoughts token usage (internal reasoning) - this can consume output tokens
		if strings.Contains(string(responseBody), "thoughtsTokenCount") {
			c.logger.Warnf("Model used internal reasoning tokens - this may consume output token budget")
		}
	}

	// Check for prompt feedback and safety issues
	if response.PromptFeedback.BlockReason != "" {
		c.logger.Warnf("Prompt was blocked: %s", response.PromptFeedback.BlockReason)
		for _, rating := range response.PromptFeedback.SafetyRatings {
			c.logger.Warnf("Safety rating - %s: %s", rating.Category, rating.Probability)
		}
		return "", fmt.Errorf("prompt blocked by safety filter: %s", response.PromptFeedback.BlockReason)
	}

	if len(response.Candidates) == 0 {
		c.logger.Warnf("No candidates in response")
		c.logger.Debugf("Full response object: %+v", response)
		return "", fmt.Errorf("no candidates in Gemini API response")
	}

	// Check the first candidate
	candidate := response.Candidates[0]
	c.logger.Debugf("First candidate finish reason: %s", candidate.FinishReason)

	if candidate.FinishReason == "SAFETY" {
		c.logger.Warnf("Content was filtered for safety reasons")
		for _, rating := range candidate.SafetyRatings {
			c.logger.Warnf("Safety rating - %s: %s", rating.Category, rating.Probability)
		}
		return "", fmt.Errorf("content filtered for safety: finish reason SAFETY")
	}

	if candidate.FinishReason == "MAX_TOKENS" {
		c.logger.Warnf("Response truncated due to token limit - consider increasing MaxOutputTokens")
		// If there's no content, it means all tokens were used for internal reasoning
		if len(candidate.Content.Parts) == 0 {
			return "", fmt.Errorf("no output generated - all tokens consumed by internal reasoning (try increasing MaxOutputTokens)")
		}
	}

	if len(candidate.Content.Parts) == 0 {
		c.logger.Warnf("No parts in first candidate")
		c.logger.Debugf("First candidate: %+v", candidate)
		return "", fmt.Errorf("no content parts in Gemini API response")
	}

	result := candidate.Content.Parts[0].Text
	c.logger.Debugf("Generated prompt with %d characters", len(result))

	return result, nil
}

// TestConnection tests the API connection and model availability with a simple prompt
func (c *Client) TestConnection(ctx context.Context) error {
	testPrompt := "Say 'Hello, this is a test' to confirm the API is working."

	c.logger.Infof("Testing Gemini API connection...")

	// Try the current configured model first
	result, err := c.GeneratePersonaPrompt(ctx, testPrompt)
	if err == nil && result != "" {
		c.logger.Infof("API connection test successful. Response: %s", result)
		return nil
	}

	c.logger.Warnf("API connection test failed: %v", err)
	return fmt.Errorf("API connection test failed: %w", err)
}

// ListAvailableModels tests the configured model (simplified since we know gemini-2.5-flash works)
func (c *Client) ListAvailableModels(ctx context.Context) {
	c.logger.Infof("Testing configured model: gemini-2.5-flash with v1 endpoint...")

	testPrompt := "Test"
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key=%s", c.apiKey)

	request := Request{
		Contents:         []Content{{Parts: []Part{{Text: testPrompt}}}},
		GenerationConfig: GenerationConfig{Temperature: 0.1, MaxOutputTokens: 10},
	}

	requestBody, _ := json.Marshal(request)
	req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(requestBody))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		c.logger.Errorf("❌ gemini-2.5-flash: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		c.logger.Infof("✅ gemini-2.5-flash: Available and working")
	} else {
		body, _ := io.ReadAll(resp.Body)
		c.logger.Errorf("❌ gemini-2.5-flash: HTTP %d - %s", resp.StatusCode, string(body))
	}
}
