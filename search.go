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
	"sort"
	"strconv"
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

type citationInsertion struct {
	offset int
	number int
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
		partOffsets := make([]int, len(candidate.Content.Parts))
		partHasText := make([]bool, len(candidate.Content.Parts))
		totalLength := 0
		for idx, part := range candidate.Content.Parts {
			if part == nil || part.Thought || part.Text == "" {
				continue
			}
			partHasText[idx] = true
			partOffsets[idx] = totalLength
			totalLength += len(part.Text)
		}

		insertions := make([]citationInsertion, 0, len(metadata.GroundingSupports))
		for _, support := range metadata.GroundingSupports {
			if support == nil || support.Segment == nil || len(support.GroundingChunkIndices) == 0 {
				continue
			}

			partIndex := int(support.Segment.PartIndex)
			if partIndex < 0 || partIndex >= len(partOffsets) || !partHasText[partIndex] {
				continue
			}

			baseOffset := partOffsets[partIndex]
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
				insertions = append(insertions, citationInsertion{
					offset: globalOffset,
					number: number,
				})
			}
		}

		if len(insertions) > 0 {
			sort.Slice(insertions, func(i, j int) bool {
				if insertions[i].offset == insertions[j].offset {
					return insertions[i].number < insertions[j].number
				}
				return insertions[i].offset < insertions[j].offset
			})

			var builder strings.Builder
			builder.Grow(len(text) + len(insertions)*4)
			lastOffset := 0
			numbers := make([]int, 0, 4)

			for i := 0; i < len(insertions); {
				offset := insertions[i].offset
				if offset < lastOffset || offset > len(text) {
					i++
					continue
				}

				builder.WriteString(text[lastOffset:offset])
				numbers = numbers[:0]

				j := i
				for ; j < len(insertions) && insertions[j].offset == offset; j++ {
					number := insertions[j].number
					if len(numbers) > 0 && numbers[len(numbers)-1] == number {
						continue
					}
					numbers = append(numbers, number)
				}
				builder.WriteString(citationText(numbers))
				lastOffset = offset
				i = j
			}

			builder.WriteString(text[lastOffset:])
			formatted = builder.String()
		}
	}

	if len(sources) == 0 {
		return formatted, nil, nil
	}

	var builder strings.Builder
	builder.Grow(len(formatted) + len(sources)*32 + len("\n\nSources:\n"))
	builder.WriteString(formatted)
	builder.WriteString("\n\nSources:\n")
	wroteSource := false
	for _, source := range sources {
		if wroteSource {
			builder.WriteByte('\n')
		}
		switch {
		case source.Title != "" && source.URI != "":
			builder.WriteByte('[')
			builder.WriteString(strconv.Itoa(source.Index))
			builder.WriteString("] ")
			builder.WriteString(source.Title)
			builder.WriteString(" (")
			builder.WriteString(source.URI)
			builder.WriteByte(')')
		case source.Title != "":
			builder.WriteByte('[')
			builder.WriteString(strconv.Itoa(source.Index))
			builder.WriteString("] ")
			builder.WriteString(source.Title)
		case source.URI != "":
			builder.WriteByte('[')
			builder.WriteString(strconv.Itoa(source.Index))
			builder.WriteString("] ")
			builder.WriteString(source.URI)
		}
		wroteSource = true
	}
	if !wroteSource {
		return formatted, sources, nil
	}

	return builder.String(), sources, nil
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
	if len(numbers) == 0 {
		return "[]"
	}

	var builder strings.Builder
	builder.Grow(len(numbers)*4 + 2)
	builder.WriteByte('[')
	for idx, number := range numbers {
		if idx > 0 {
			builder.WriteByte(',')
		}
		builder.WriteString(strconv.Itoa(number))
	}
	builder.WriteByte(']')
	return builder.String()
}
