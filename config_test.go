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
	"testing"

	"google.golang.org/genai"
)

func TestLoadConfigFromEnv(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		env     map[string]string
		want    serverConfig
		wantErr string
	}{
		{
			name: "google api key",
			env: map[string]string{
				GoogleAPIKey:   "google-key",
				EnvGeminiModel: "gemini-2.0-flash",
			},
			want: serverConfig{
				Model: "gemini-2.0-flash",
				ClientConfig: &genai.ClientConfig{
					Backend: genai.BackendGeminiAPI,
					APIKey:  "google-key",
				},
			},
		},
		{
			name: "gemini api key fallback and custom model",
			env: map[string]string{
				GoogleAPIKey: "test-key",
			},
			want: serverConfig{
				Model: DefaultModel,
				ClientConfig: &genai.ClientConfig{
					Backend: genai.BackendGeminiAPI,
					APIKey:  "test-key",
				},
			},
		},
		{
			name: "vertex provider compatibility env",
			env: map[string]string{
				EnvGoogleCloudProject:     "project-1",
				EnvGoogleGenAIUseVertexAI: "true",
			},
			want: serverConfig{
				Model: DefaultModel,
				ClientConfig: &genai.ClientConfig{
					Backend:  genai.BackendVertexAI,
					Project:  "project-1",
					Location: DefaultLocation,
				},
			},
		},
		{
			name: "vertex native go sdk env fallback",
			env: map[string]string{
				EnvGoogleCloudProject:     "project-2",
				EnvGoogleCloudLocation:    "asia-northeast1",
				EnvGoogleGenAIUseVertexAI: "true",
			},
			want: serverConfig{
				Model: DefaultModel,
				ClientConfig: &genai.ClientConfig{
					Backend:  genai.BackendVertexAI,
					Project:  "project-2",
					Location: "asia-northeast1",
				},
			},
		},
		{
			name:    "missing api key",
			env:     map[string]string{},
			wantErr: "\"GOOGLE_API_KEY\" or \"GEMINI_API_KEY\" environment variable is required when using Google AI Studio",
		},
		{
			name: "missing vertex project",
			env: map[string]string{
				EnvGoogleGenAIUseVertexAI: "true",
			},
			wantErr: "\"GOOGLE_CLOUD_PROJECT\" environment variable is required when using Google Vertex AI",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := loadConfigFromEnv(func(key string) string {
				return tt.env[key]
			})
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("loadConfigFromEnv() error = nil, want %q", tt.wantErr)
				}
				if err.Error() != tt.wantErr {
					t.Fatalf("loadConfigFromEnv() error = %q, want %q", err.Error(), tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("loadConfigFromEnv() error = %v", err)
			}
			if got.Model != tt.want.Model {
				t.Fatalf("loadConfigFromEnv() model = %q, want %q", got.Model, tt.want.Model)
			}
			if got.ClientConfig.Backend != tt.want.ClientConfig.Backend {
				t.Fatalf("loadConfigFromEnv() backend = %v, want %v", got.ClientConfig.Backend, tt.want.ClientConfig.Backend)
			}
			if got.ClientConfig.APIKey != tt.want.ClientConfig.APIKey {
				t.Fatalf("loadConfigFromEnv() api key = %q, want %q", got.ClientConfig.APIKey, tt.want.ClientConfig.APIKey)
			}
			if got.ClientConfig.Project != tt.want.ClientConfig.Project {
				t.Fatalf("loadConfigFromEnv() project = %q, want %q", got.ClientConfig.Project, tt.want.ClientConfig.Project)
			}
			if got.ClientConfig.Location != tt.want.ClientConfig.Location {
				t.Fatalf("loadConfigFromEnv() location = %q, want %q", got.ClientConfig.Location, tt.want.ClientConfig.Location)
			}
		})
	}
}

func TestServerConfigNewClientRejectsMutuallyExclusiveSettings(t *testing.T) {
	t.Parallel()

	cfg := serverConfig{
		Model: DefaultModel,
		ClientConfig: &genai.ClientConfig{
			Backend: genai.BackendGeminiAPI,
			APIKey:  "test-key",
			Project: "project-1",
		},
	}

	_, err := cfg.NewClient(context.Background())
	if err == nil {
		t.Fatal("newClient() error = nil, want non-nil")
	}
}

func TestFirstNonEmpty(t *testing.T) {
	t.Parallel()

	got := firstNonEmpty("", "  ", "value", "other")
	if got != "value" {
		t.Fatalf("firstNonEmpty() = %q, want %q", got, "value")
	}
}

func TestIsTruthy(t *testing.T) {
	t.Parallel()

	truthy := []string{"1", "true", "TRUE", " yes ", "on"}
	for _, value := range truthy {
		if !isEnabled(value) {
			t.Fatalf("isTruthy(%q) = false, want true", value)
		}
	}

	falsy := []string{"", "0", "false", "off"}
	for _, value := range falsy {
		if isEnabled(value) {
			t.Fatalf("isTruthy(%q) = true, want false", value)
		}
	}
}
