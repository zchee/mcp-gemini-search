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
	"errors"
	"testing"

	"google.golang.org/genai"
)

type stubGenerator struct {
	resp        *genai.GenerateContentResponse
	err         error
	gotModel    string
	gotContents []*genai.Content
	gotConfig   *genai.GenerateContentConfig
}

func (s *stubGenerator) GenerateContent(_ context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
	s.gotModel = model
	s.gotContents = contents
	s.gotConfig = config
	return s.resp, s.err
}

func TestGoogleSearchServiceSearch(t *testing.T) {
	t.Parallel()

	stub := &stubGenerator{
		resp: &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{Text: "Answer"},
						},
					},
					GroundingMetadata: &genai.GroundingMetadata{
						GroundingChunks: []*genai.GroundingChunk{
							{Web: &genai.GroundingChunkWeb{Title: "Example", URI: "https://example.com"}},
						},
						GroundingSupports: []*genai.GroundingSupport{
							{
								Segment:               &genai.Segment{PartIndex: 0, EndIndex: 6},
								GroundingChunkIndices: []int32{0},
							},
						},
					},
				},
			},
		},
	}
	svc := &googleSearchService{
		model:     "gemini-2.5-flash",
		generator: stub,
	}

	got, err := svc.Search(context.Background(), "golang")
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if got.Query != "golang" {
		t.Fatalf("Search() query = %q, want %q", got.Query, "golang")
	}
	if got.Text != "Answer[1]\n\nSources:\n[1] Example (https://example.com)" {
		t.Fatalf("Search() text = %q", got.Text)
	}
	if len(got.Sources) != 1 {
		t.Fatalf("Search() sources len = %d, want 1", len(got.Sources))
	}
	if stub.gotModel != "gemini-2.5-flash" {
		t.Fatalf("GenerateContent() model = %q, want %q", stub.gotModel, "gemini-2.5-flash")
	}
	if len(stub.gotContents) != 1 || len(stub.gotContents[0].Parts) != 1 || stub.gotContents[0].Parts[0].Text != "golang" {
		t.Fatalf("GenerateContent() contents = %#v, want single user text part", stub.gotContents)
	}
	if stub.gotConfig == nil || len(stub.gotConfig.Tools) != 1 || stub.gotConfig.Tools[0].GoogleSearch == nil {
		t.Fatalf("GenerateContent() config = %#v, want google search tool", stub.gotConfig)
	}
}

func TestGoogleSearchServiceSearchErrors(t *testing.T) {
	t.Parallel()

	svc := &googleSearchService{model: "gemini-2.5-flash"}

	if _, err := svc.Search(context.Background(), "golang"); err == nil || err.Error() != "google search service is not configured" {
		t.Fatalf("Search() nil generator error = %v, want google search service is not configured", err)
	}

	svc.generator = &stubGenerator{}

	if _, err := svc.Search(context.Background(), "   "); err == nil || err.Error() != "search query cannot be empty" {
		t.Fatalf("Search() empty query error = %v, want search query cannot be empty", err)
	}

	backendErr := errors.New("backend failed")
	svc.generator = &stubGenerator{err: backendErr}
	if _, err := svc.Search(context.Background(), "golang"); err == nil || err.Error() != "google search failed: backend failed" {
		t.Fatalf("Search() backend error = %v, want wrapped backend error", err)
	}
}

func TestFormatGroundedResponse(t *testing.T) {
	t.Parallel()

	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{Text: "Alpha "},
						{Text: "Beta"},
					},
				},
				GroundingMetadata: &genai.GroundingMetadata{
					GroundingChunks: []*genai.GroundingChunk{
						{Web: &genai.GroundingChunkWeb{Title: "First", URI: "https://first.example"}},
						{Maps: &genai.GroundingChunkMaps{Title: "Second", URI: "https://second.example"}},
					},
					GroundingSupports: []*genai.GroundingSupport{
						{
							Segment:               &genai.Segment{PartIndex: 0, EndIndex: 6},
							GroundingChunkIndices: []int32{0},
						},
						{
							Segment:               &genai.Segment{PartIndex: 1, EndIndex: 4},
							GroundingChunkIndices: []int32{0, 1},
						},
					},
				},
			},
		},
	}

	gotText, gotSources, err := formatGroundedResponse(resp)
	if err != nil {
		t.Fatalf("formatGroundedResponse() error = %v", err)
	}

	wantText := "Alpha [1]Beta[1,2]\n\nSources:\n[1] First (https://first.example)\n[2] Second (https://second.example)"
	if gotText != wantText {
		t.Fatalf("formatGroundedResponse() text = %q, want %q", gotText, wantText)
	}
	if len(gotSources) != 2 {
		t.Fatalf("formatGroundedResponse() sources len = %d, want 2", len(gotSources))
	}
	if gotSources[1].Title != "Second" || gotSources[1].URI != "https://second.example" {
		t.Fatalf("formatGroundedResponse() sources[1] = %#v", gotSources[1])
	}
}

func TestFormatGroundedResponseNoText(t *testing.T) {
	t.Parallel()

	_, _, err := formatGroundedResponse(&genai.GenerateContentResponse{})
	if err == nil || err.Error() != "no response from Gemini model" {
		t.Fatalf("formatGroundedResponse() error = %v, want no response from Gemini model", err)
	}
}

func TestFormatGroundedResponseOrdersAndDeduplicatesCitations(t *testing.T) {
	t.Parallel()

	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{Text: "Alpha "},
						{Text: "Beta"},
					},
				},
				GroundingMetadata: &genai.GroundingMetadata{
					GroundingChunks: []*genai.GroundingChunk{
						{Web: &genai.GroundingChunkWeb{Title: "One", URI: "https://one.example"}},
						{Web: &genai.GroundingChunkWeb{Title: "Two", URI: "https://two.example"}},
					},
					GroundingSupports: []*genai.GroundingSupport{
						{
							Segment:               &genai.Segment{PartIndex: 1, EndIndex: 4},
							GroundingChunkIndices: []int32{1, 0, 1},
						},
						{
							Segment:               &genai.Segment{PartIndex: 0, EndIndex: 6},
							GroundingChunkIndices: []int32{0},
						},
						{
							Segment:               &genai.Segment{PartIndex: 1, EndIndex: 4},
							GroundingChunkIndices: []int32{0},
						},
					},
				},
			},
		},
	}

	gotText, gotSources, err := formatGroundedResponse(resp)
	if err != nil {
		t.Fatalf("formatGroundedResponse() error = %v", err)
	}

	wantText := "Alpha [1]Beta[1,2]\n\nSources:\n[1] One (https://one.example)\n[2] Two (https://two.example)"
	if gotText != wantText {
		t.Fatalf("formatGroundedResponse() text = %q, want %q", gotText, wantText)
	}
	if len(gotSources) != 2 {
		t.Fatalf("formatGroundedResponse() sources len = %d, want 2", len(gotSources))
	}
}

func TestGroundingSource(t *testing.T) {
	t.Parallel()

	title, uri := groundingSource(&genai.GroundingChunk{
		Image: &genai.GroundingChunkImage{
			Title:     "Image Result",
			SourceURI: "https://source.example",
			ImageURI:  "https://image.example",
		},
	})
	if title != "Image Result" || uri != "https://source.example" {
		t.Fatalf("groundingSource() = (%q, %q), want (%q, %q)", title, uri, "Image Result", "https://source.example")
	}
}

func TestCitationText(t *testing.T) {
	t.Parallel()

	if got := citationText([]int{1, 2, 3}); got != "[1,2,3]" {
		t.Fatalf("citationText() = %q, want %q", got, "[1,2,3]")
	}
}
