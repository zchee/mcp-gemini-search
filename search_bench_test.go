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

var benchmarkGroundedResponse = &genai.GenerateContentResponse{
	Candidates: []*genai.Candidate{
		{
			Content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "Alpha "},
					{Text: "Beta "},
					{Text: "Gamma Delta Epsilon Zeta Eta Theta"},
				},
			},
			GroundingMetadata: &genai.GroundingMetadata{
				GroundingChunks: []*genai.GroundingChunk{
					{Web: &genai.GroundingChunkWeb{Title: "First", URI: "https://first.example"}},
					{Maps: &genai.GroundingChunkMaps{Title: "Second", URI: "https://second.example"}},
					{RetrievedContext: &genai.GroundingChunkRetrievedContext{Title: "Third", URI: "https://third.example"}},
					{Image: &genai.GroundingChunkImage{Title: "Fourth", SourceURI: "https://fourth.example"}},
				},
				GroundingSupports: []*genai.GroundingSupport{
					{
						Segment:               &genai.Segment{PartIndex: 0, EndIndex: 6},
						GroundingChunkIndices: []int32{0},
					},
					{
						Segment:               &genai.Segment{PartIndex: 1, EndIndex: 5},
						GroundingChunkIndices: []int32{0, 1},
					},
					{
						Segment:               &genai.Segment{PartIndex: 2, EndIndex: 5},
						GroundingChunkIndices: []int32{2},
					},
					{
						Segment:               &genai.Segment{PartIndex: 2, EndIndex: 13},
						GroundingChunkIndices: []int32{1, 2, 3},
					},
					{
						Segment:               &genai.Segment{PartIndex: 2, EndIndex: 21},
						GroundingChunkIndices: []int32{3},
					},
					{
						Segment:               &genai.Segment{PartIndex: 2, EndIndex: 26},
						GroundingChunkIndices: []int32{0, 3},
					},
					{
						Segment:               &genai.Segment{PartIndex: 2, EndIndex: 30},
						GroundingChunkIndices: []int32{1},
					},
					{
						Segment:               &genai.Segment{PartIndex: 2, EndIndex: 39},
						GroundingChunkIndices: []int32{0, 2, 3},
					},
				},
			},
		},
	},
}

func BenchmarkFormatGroundedResponse(b *testing.B) {
	b.ReportAllocs()

	for b.Loop() {
		if _, _, err := formatGroundedResponse(benchmarkGroundedResponse); err != nil {
			b.Fatalf("formatGroundedResponse() error = %v", err)
		}
	}
}

func BenchmarkGoogleSearchServiceSearch(b *testing.B) {
	b.ReportAllocs()

	svc := &googleSearchService{
		model: "gemini-2.5-flash",
		generator: &stubGenerator{
			resp: benchmarkGroundedResponse,
		},
	}
	ctx := context.Background()

	for b.Loop() {
		if _, err := svc.Search(ctx, "latest golang release notes"); err != nil {
			b.Fatalf("Search() error = %v", err)
		}
	}
}
