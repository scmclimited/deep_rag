#!/bin/bash
# Quick test script - tests one example of each endpoint type
# Useful for quick verification after changes

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000"
# Use forward slashes for Docker paths (works on both Windows and Linux)
SAMPLE_PDF="inference/samples/NYMBL - AI Engineer - Omar.pdf"
SAMPLE_IMAGE="inference/samples/technical_assessment_brief_1.png"

# Check if services are running
if ! curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: API is not running. Please run 'make up' first.${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Quick Endpoint Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Health check
echo -e "${GREEN}[1] Health Check${NC}"
curl -s "$API_URL/health" | jq '.' || curl -s "$API_URL/health"
echo ""
echo ""

# Ingest
echo -e "${GREEN}[2] Ingest PDF${NC}"
RESPONSE=$(curl -s -X POST "$API_URL/ingest" \
    -F "attachment=@$SAMPLE_PDF" \
    -F "title=Quick Test PDF")
echo "$RESPONSE" | jq '.' || echo "$RESPONSE"
DOC_ID=$(echo "$RESPONSE" | jq -r '.doc_id // empty' || echo "")
echo ""
echo ""

# Query (if we got a doc_id)
if [ -n "$DOC_ID" ] && [ "$DOC_ID" != "null" ]; then
    echo -e "${GREEN}[3] Query with doc_id${NC}"
    curl -s -X POST "$API_URL/ask" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"What is this about?\", \"doc_id\": \"$DOC_ID\"}" | jq '.' || curl -s -X POST "$API_URL/ask" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"What is this about?\", \"doc_id\": \"$DOC_ID\"}"
    echo ""
    echo ""
fi

# Query-graph
echo -e "${GREEN}[4] Query-graph (LangGraph)${NC}"
curl -s -X POST "$API_URL/ask-graph" \
    -H "Content-Type: application/json" \
    -d '{"question": "What are the key points?", "thread_id": "quick-test"}' | jq '.' || curl -s -X POST "$API_URL/ask-graph" \
    -H "Content-Type: application/json" \
    -d '{"question": "What are the key points?", "thread_id": "quick-test"}'
echo ""
echo ""

# Infer
echo -e "${GREEN}[5] Infer (ingest + query)${NC}"
curl -s -X POST "$API_URL/infer" \
    -F "question=What does this say?" \
    -F "attachment=@$SAMPLE_IMAGE" \
    -F "title=Quick Test Image" | jq '.' || curl -s -X POST "$API_URL/infer" \
    -F "question=What does this say?" \
    -F "attachment=@$SAMPLE_IMAGE" \
    -F "title=Quick Test Image"
echo ""
echo ""

# Infer-graph
echo -e "${GREEN}[6] Infer-graph (LangGraph)${NC}"
curl -s -X POST "$API_URL/infer-graph" \
    -F "question=What is in this image?" \
    -F "attachment=@$SAMPLE_IMAGE" \
    -F "title=Quick Test Graph Image" \
    -F "thread_id=quick-test" | jq '.' || curl -s -X POST "$API_URL/infer-graph" \
    -F "question=What is in this image?" \
    -F "attachment=@$SAMPLE_IMAGE" \
    -F "title=Quick Test Graph Image" \
    -F "thread_id=quick-test"
echo ""
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Quick test completed${NC}"
echo -e "${BLUE}========================================${NC}"

