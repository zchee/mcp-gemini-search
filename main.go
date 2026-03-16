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

// Command mcp-gemini-google-search provides Google Search functionality using
// Gemini's built-in Grounding with Google Search feature.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/google/uuid"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/zchee/dumper"
)

const (
	toolName        = "google_search"
	defaultLogLevel = slog.LevelDebug
)

var Version = "0.0.1"

var flagLogPath string

func init() {
	uuid.EnableRandPool()

	flag.StringVar(&flagLogPath, "logpath", "", "if set, enable MCP server logging")
}

type googleSearchParams struct {
	Query string `json:"query" jsonschema:"The search query to find information on the web."`
}

type googleSearchSource struct {
	Index int    `json:"index" jsonschema:"The 1-based citation index shown in the response text."`
	Title string `json:"title,omitempty" jsonschema:"The title of the cited source."`
	URI   string `json:"uri,omitempty" jsonschema:"The source URL or canonical URI."`
}

type googleSearchOutput struct {
	Query   string               `json:"query" jsonschema:"The original search query."`
	Text    string               `json:"text" jsonschema:"The grounded response text with inline citations and an appended source list when available."`
	Sources []googleSearchSource `json:"sources,omitempty" jsonschema:"The sources referenced by the grounded response."`
}

func main() {
	flag.Parse()

	dumper.Config = dumper.ConfigState{
		Indent:       " ",
		NumericWidth: 1,
		StringWidth:  1,
		BytesWidth:   16,
		CommentBytes: true,
		OmitZero:     true,
	}

	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	var logWriter io.WriteCloser

	handler := slog.DiscardHandler
	if flagLogPath != "" {
		file, err := os.OpenFile(flagLogPath, os.O_RDWR|os.O_CREATE, 0o666)
		if err != nil {
			return fmt.Errorf("open %q file: %w", flagLogPath, err)
		}
		logWriter = file
		defer logWriter.Close()

		handler = slog.NewTextHandler(logWriter, &slog.HandlerOptions{
			Level: defaultLogLevel,
		})
	}
	logger := slog.New(handler)
	slog.SetDefault(logger)

	cfg, err := loadConfigFromEnv(os.Getenv)
	if err != nil {
		return err
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	client, err := cfg.newClient(ctx)
	if err != nil {
		return fmt.Errorf("create genai client: %w", err)
	}
	searchService := &googleSearchService{
		model:     cfg.Model,
		generator: client.Models,
	}

	srvImpl := &mcp.Implementation{
		Name:       "mcp-gemini-google-search",
		Version:    Version,
		WebsiteURL: "https://github.com/zchee/mcp-gemini-google-search",
	}
	opts := &mcp.ServerOptions{
		Logger:   logger,
		HasTools: true,
		GetSessionID: func() string {
			return uuid.Must(uuid.NewV7()).String()
		},
		Capabilities: &mcp.ServerCapabilities{
			Tools: &mcp.ToolCapabilities{},
		},
	}
	srv := mcp.NewServer(srvImpl, opts)

	googleSearchTool := &mcp.Tool{
		Name:        toolName,
		Description: "Performs a web search using Google Search (via the Gemini API) and returns the results. This tool is useful for finding information on the internet based on a query.",
	}
	googleSearchToolHandler := func(ctx context.Context, _ *mcp.CallToolRequest, input googleSearchParams) (*mcp.CallToolResult, googleSearchOutput, error) {
		output, err := searchService.Search(ctx, input.Query)
		if err != nil {
			return nil, googleSearchOutput{}, err
		}

		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.TextContent{Text: output.Text},
			},
		}, output, nil
	}
	mcp.AddTool(srv, googleSearchTool, googleSearchToolHandler)

	transport := mcp.Transport(&mcp.StdioTransport{})
	if logWriter != nil {
		transport = &mcp.LoggingTransport{
			Transport: transport,
			Writer:    logWriter,
		}
	}

	logger.InfoContext(ctx, "gemini google search mcp server running on stdio")
	if err := srv.Run(ctx, transport); err != nil {
		logger.ErrorContext(ctx, "serve gemini google search mcp stdio server", slog.Any("error", err))
		return fmt.Errorf("serve gemini google search mcp stdio server: %w", err)
	}

	return nil
}
