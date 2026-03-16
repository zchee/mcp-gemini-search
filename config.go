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
	defaultModel    = "gemini-3.1-pro-preview"
	defaultLocation = "global"
)

type serverConfig struct {
	Model        string
	ClientConfig genai.ClientConfig
}

func loadConfigFromEnv(getenv func(string) string) (serverConfig, error) {
	provider := strings.TrimSpace(getenv("GEMINI_PROVIDER"))
	useVertex := provider == "vertex" || isTruthy(getenv("GOOGLE_GENAI_USE_VERTEXAI"))
	if provider != "" && provider != "vertex" {
		return serverConfig{}, fmt.Errorf("unsupported GEMINI_PROVIDER %q", provider)
	}

	cfg := serverConfig{
		Model: strings.TrimSpace(getenv("GEMINI_MODEL")),
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}

	if useVertex {
		project := firstNonEmpty(
			getenv("VERTEX_PROJECT_ID"),
			getenv("GOOGLE_CLOUD_PROJECT"),
		)
		if project == "" {
			return serverConfig{}, fmt.Errorf("VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT environment variable is required when using Vertex AI")
		}

		location := firstNonEmpty(
			getenv("VERTEX_LOCATION"),
			getenv("GOOGLE_CLOUD_LOCATION"),
			getenv("GOOGLE_CLOUD_REGION"),
		)
		if location == "" {
			location = defaultLocation
		}

		cfg.ClientConfig = genai.ClientConfig{
			Backend:  genai.BackendVertexAI,
			Project:  project,
			Location: location,
		}
		return cfg, nil
	}

	apiKey := firstNonEmpty(
		getenv("GEMINI_API_KEY"),
		getenv("GOOGLE_API_KEY"),
	)
	if apiKey == "" {
		return serverConfig{}, fmt.Errorf("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required when using Google AI Studio")
	}

	cfg.ClientConfig = genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  apiKey,
	}
	return cfg, nil
}

func (c serverConfig) newClient(ctx context.Context) (*genai.Client, error) {
	return genai.NewClient(ctx, &c.ClientConfig)
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

func isTruthy(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
