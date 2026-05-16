// Copyright 2026 The mcp-gemini-google-search Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/genai"
)

const (
	// EnvGeminiModel is the environment variable for the Google Gemini model to use (e.g., "gemini-3.1-pro-preview").
	// If not set, the server will use a default model.
	EnvGeminiModel = "GEMINI_MODEL"

	// GeminiAPIKey and GoogleAPIKey are environment variables for the API key to use when connecting to the Google Gemini API.
	// If bot h are set, [GoogleAPIKey] will take precedence over [GeminiAPIKey].
	GoogleAPIKey = "GOOGLE_API_KEY"
	GeminiAPIKey = "GEMINI_API_KEY"

	// EnvGoogleCloudProject is the environment variable for the Google Cloud project ID, used when connecting to Vertex AI.
	EnvGoogleCloudProject = "GOOGLE_CLOUD_PROJECT"

	// EnvGoogleCloudLocation is the environment variable for the Google Cloud location/region, used when connecting to Vertex AI.
	EnvGoogleCloudLocation = "GOOGLE_CLOUD_LOCATION"

	// EnvGoogleGenAIUseVertexAI is the environment variable to indicate whether to use Vertex AI as the backend for Google GenAI.
	// If set to a truthy value (e.g., "1", "true", "yes"), the server will use Vertex AI; otherwise, it will use the Google Gemini API.
	EnvGoogleGenAIUseVertexAI = "GOOGLE_GENAI_USE_VERTEXAI"
)

const (
	// DefaultModel is the default Google Gemini model to use if none is specified in the environment variables.
	DefaultModel = "gemini-3.1-pro-preview"
	// DefaultLocation is the default location to use for Google Cloud Vertex AI if none is specified in the environment variables.
	DefaultLocation = "global"
)

type serverConfig struct {
	Model        string
	ClientConfig *genai.ClientConfig
}

func loadConfigFromEnv(getenv func(string) string) (serverConfig, error) {
	cfg := serverConfig{
		Model: strings.TrimSpace(getenv(EnvGeminiModel)),
	}
	if cfg.Model == "" {
		cfg.Model = DefaultModel
	}

	useVertex := isEnabled(getenv(EnvGoogleGenAIUseVertexAI))
	if useVertex {
		project := getenv(EnvGoogleCloudProject)
		if project == "" {
			return serverConfig{}, fmt.Errorf("%q environment variable is required when using Google Vertex AI", EnvGoogleCloudProject)
		}

		location := getenv(EnvGoogleCloudLocation)
		if location == "" {
			location = DefaultLocation
		}

		cfg.ClientConfig = &genai.ClientConfig{
			Backend:  genai.BackendVertexAI,
			Project:  project,
			Location: location,
		}
		return cfg, nil
	}

	apiKey := firstNonEmpty(getenv(GoogleAPIKey), getenv(GeminiAPIKey))
	if apiKey == "" {
		return serverConfig{}, fmt.Errorf("%q or %q environment variable is required when using Google AI Studio", GoogleAPIKey, GeminiAPIKey)
	}

	cfg.ClientConfig = &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  apiKey,
	}
	return cfg, nil
}

// NewClient creates a new [*genai.Client] based on the server configuration.
func (c *serverConfig) NewClient(ctx context.Context) (*genai.Client, error) {
	return genai.NewClient(ctx, c.ClientConfig)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}

func isEnabled(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
