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
	"slices"
	"sort"
	"strings"

	"google.golang.org/genai"
)

type contentGenerator interface {
	GenerateContent(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error)
}

type googleSearchService struct {
	model     string
	generator contentGenerator
}

func (s *googleSearchService) Search(ctx context.Context, query string) (googleSearchOutput, error) {
	if s.generator == nil {
		return googleSearchOutput{}, fmt.Errorf("google search service is not configured")
	}
	if strings.TrimSpace(query) == "" {
		return googleSearchOutput{}, fmt.Errorf("search query cannot be empty")
	}

	resp, err := s.generator.GenerateContent(ctx, s.model, genai.Text(query), &genai.GenerateContentConfig{
		Tools: []*genai.Tool{
			{GoogleSearch: &genai.GoogleSearch{}},
		},
	})
	if err != nil {
		return googleSearchOutput{}, fmt.Errorf("google search failed: %w", err)
	}

	text, sources, err := formatGroundedResponse(resp)
	if err != nil {
		return googleSearchOutput{}, fmt.Errorf("google search failed: %w", err)
	}

	return googleSearchOutput{
		Query:   query,
		Text:    text,
		Sources: sources,
	}, nil
}

func formatGroundedResponse(resp *genai.GenerateContentResponse) (string, []googleSearchSource, error) {
	if resp == nil {
		return "", nil, fmt.Errorf("no response from Gemini model")
	}

	text := resp.Text()
	if strings.TrimSpace(text) == "" {
		return "", nil, fmt.Errorf("no response from Gemini model")
	}

	if len(resp.Candidates) == 0 || resp.Candidates[0] == nil {
		return text, nil, nil
	}

	candidate := resp.Candidates[0]
	metadata := candidate.GroundingMetadata
	if metadata == nil {
		return text, nil, nil
	}

	sources := make([]googleSearchSource, 0, len(metadata.GroundingChunks))
	for idx, chunk := range metadata.GroundingChunks {
		title, uri := groundingSource(chunk)
		if title == "" && uri == "" {
			continue
		}
		sources = append(sources, googleSearchSource{
			Index: idx + 1,
			Title: title,
			URI:   uri,
		})
	}

	formatted := text
	if candidate.Content != nil && len(candidate.Content.Parts) > 0 && len(metadata.GroundingSupports) > 0 && len(sources) > 0 {
		partOffsets := make(map[int]int, len(candidate.Content.Parts))
		totalLength := 0
		for idx, part := range candidate.Content.Parts {
			if part == nil || part.Thought || part.Text == "" {
				continue
			}
			partOffsets[idx] = totalLength
			totalLength += len(part.Text)
		}

		insertions := make(map[int][]int)
		for _, support := range metadata.GroundingSupports {
			if support == nil || support.Segment == nil || len(support.GroundingChunkIndices) == 0 {
				continue
			}

			partIndex := int(support.Segment.PartIndex)
			baseOffset, ok := partOffsets[partIndex]
			if !ok {
				continue
			}

			partText := candidate.Content.Parts[partIndex].Text
			endIndex := int(support.Segment.EndIndex)
			if endIndex < 0 || endIndex > len(partText) {
				continue
			}

			globalOffset := baseOffset + endIndex
			for _, chunkIndex := range support.GroundingChunkIndices {
				number := int(chunkIndex) + 1
				if number <= 0 {
					continue
				}
				insertions[globalOffset] = append(insertions[globalOffset], number)
			}
		}

		if len(insertions) > 0 {
			offsets := make([]int, 0, len(insertions))
			for offset := range insertions {
				offsets = append(offsets, offset)
			}
			sort.Sort(sort.Reverse(sort.IntSlice(offsets)))

			for _, offset := range offsets {
				numbers := insertions[offset]
				slices.Sort(numbers)
				numbers = slices.Compact(numbers)
				if len(numbers) == 0 {
					continue
				}
				formatted = formatted[:offset] + citationText(numbers) + formatted[offset:]
			}
		}
	}

	if len(sources) == 0 {
		return formatted, nil, nil
	}

	var sourceLines []string
	for _, source := range sources {
		switch {
		case source.Title != "" && source.URI != "":
			sourceLines = append(sourceLines, fmt.Sprintf("[%d] %s (%s)", source.Index, source.Title, source.URI))
		case source.Title != "":
			sourceLines = append(sourceLines, fmt.Sprintf("[%d] %s", source.Index, source.Title))
		case source.URI != "":
			sourceLines = append(sourceLines, fmt.Sprintf("[%d] %s", source.Index, source.URI))
		}
	}
	if len(sourceLines) == 0 {
		return formatted, sources, nil
	}

	return formatted + "\n\nSources:\n" + strings.Join(sourceLines, "\n"), sources, nil
}

func groundingSource(chunk *genai.GroundingChunk) (string, string) {
	if chunk == nil {
		return "", ""
	}
	switch {
	case chunk.Web != nil:
		return chunk.Web.Title, chunk.Web.URI
	case chunk.Maps != nil:
		return chunk.Maps.Title, chunk.Maps.URI
	case chunk.RetrievedContext != nil:
		return chunk.RetrievedContext.Title, chunk.RetrievedContext.URI
	case chunk.Image != nil:
		return chunk.Image.Title, firstNonEmpty(chunk.Image.SourceURI, chunk.Image.ImageURI)
	default:
		return "", ""
	}
}

func citationText(numbers []int) string {
	var parts []string
	for _, number := range numbers {
		parts = append(parts, fmt.Sprintf("%d", number))
	}
	return "[" + strings.Join(parts, ",") + "]"
}
