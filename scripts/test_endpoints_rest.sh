#!/bin/bash
# Comprehensive test script for all REST API endpoints
# Tests ingest, query, and infer endpoints with various configurations

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000"
# Use forward slashes for Docker paths (works on both Windows and Linux)
SAMPLE_PDF="inference/samples/NYMBL - AI Engineer - Omar.pdf"
SAMPLE_IMAGE="inference/samples/technical_assessment_brief_1.png"
SAMPLE_IMAGE2="inference/samples/technical_assessment_brief_2.png"

# Check if services are running
if ! curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: API is not running. Please run 'make up' first.${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Deep RAG Endpoint Testing (REST API)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Store doc_id for subsequent queries
DOC_ID=""
THREAD_ID="test-thread-$(date +%s)"

# =============================================================================
# 0. HEALTH CHECK
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}0. HEALTH CHECK${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}[0.1] Health Check${NC}"
echo "Command: curl -X GET $API_URL/health"
curl -s -X GET "$API_URL/health" | jq '.' || curl -s -X GET "$API_URL/health"
echo ""
echo ""

# =============================================================================
# 1. INGEST ENDPOINTS
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}1. TESTING INGEST ENDPOINTS${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}[1.1] Ingest PDF${NC}"
echo "Command: curl -X POST $API_URL/ingest -F \"attachment=@$SAMPLE_PDF\" -F \"title=Test PDF Document\""
RESPONSE=$(curl -s -X POST "$API_URL/ingest" \
    -F "attachment=@$SAMPLE_PDF" \
    -F "title=Test PDF Document")
echo "$RESPONSE" | jq '.' || echo "$RESPONSE"
DOC_ID=$(echo "$RESPONSE" | jq -r '.doc_id // empty' || echo "")
if [ -n "$DOC_ID" ] && [ "$DOC_ID" != "null" ]; then
    echo -e "${GREEN}✓ Document ingested with doc_id: $DOC_ID${NC}"
else
    echo -e "${YELLOW}⚠ Could not extract doc_id from response${NC}"
fi
echo ""

echo -e "${GREEN}[1.2] Ingest Image${NC}"
echo "Command: curl -X POST $API_URL/ingest -F \"attachment=@$SAMPLE_IMAGE\" -F \"title=Test Image Document\""
RESPONSE=$(curl -s -X POST "$API_URL/ingest" \
    -F "attachment=@$SAMPLE_IMAGE" \
    -F "title=Test Image Document")
echo "$RESPONSE" | jq '.' || echo "$RESPONSE"
echo ""

# =============================================================================
# 2. ASK ENDPOINTS (Direct Pipeline)
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}2. TESTING ASK ENDPOINTS (Direct Pipeline)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}[2.1] Ask all documents (no doc_id)${NC}"
echo "Command: curl -X POST $API_URL/ask -H \"Content-Type: application/json\" -d '{\"question\": \"What is this document about?\"}'"
curl -s -X POST "$API_URL/ask" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is this document about?"}' | jq '.' || curl -s -X POST "$API_URL/ask" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is this document about?"}'
echo ""
echo ""

if [ -n "$DOC_ID" ] && [ "$DOC_ID" != "null" ]; then
    echo -e "${GREEN}[2.2] Ask specific document (with doc_id)${NC}"
    echo "Command: curl -X POST $API_URL/ask -H \"Content-Type: application/json\" -d '{\"question\": \"What are the main requirements?\", \"doc_id\": \"$DOC_ID\"}'"
    curl -s -X POST "$API_URL/ask" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"What are the main requirements?\", \"doc_id\": \"$DOC_ID\"}" | jq '.' || curl -s -X POST "$API_URL/ask" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"What are the main requirements?\", \"doc_id\": \"$DOC_ID\"}"
    echo ""
    echo ""
    
    echo -e "${GREEN}[2.3] Ask with cross-doc enabled (with doc_id)${NC}"
    echo "Command: curl -X POST $API_URL/ask -H \"Content-Type: application/json\" -d '{\"question\": \"What are the technical requirements?\", \"doc_id\": \"$DOC_ID\", \"cross_doc\": true}'"
    curl -s -X POST "$API_URL/ask" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"What are the technical requirements?\", \"doc_id\": \"$DOC_ID\", \"cross_doc\": true}" | jq '.' || curl -s -X POST "$API_URL/ask" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"What are the technical requirements?\", \"doc_id\": \"$DOC_ID\", \"cross_doc\": true}"
    echo ""
    echo ""
else
    echo -e "${YELLOW}[2.2-2.3] Skipping doc_id-specific queries (no doc_id available)${NC}"
    echo ""
fi

echo -e "${GREEN}[2.4] Ask all documents with cross-doc${NC}"
echo "Command: curl -X POST $API_URL/ask -H \"Content-Type: application/json\" -d '{\"question\": \"What are the key points?\", \"cross_doc\": true}'"
curl -s -X POST "$API_URL/ask" \
    -H "Content-Type: application/json" \
    -d '{"question": "What are the key points?", "cross_doc": true}' | jq '.' || curl -s -X POST "$API_URL/ask" \
    -H "Content-Type: application/json" \
    -d '{"question": "What are the key points?", "cross_doc": true}'
echo ""
echo ""

# =============================================================================
# 3. ASK-GRAPH ENDPOINTS (LangGraph Pipeline)
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}3. TESTING ASK-GRAPH ENDPOINTS (LangGraph Pipeline)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}[3.1] Ask-graph all documents (default thread_id)${NC}"
echo "Command: curl -X POST $API_URL/ask-graph -H \"Content-Type: application/json\" -d '{\"question\": \"What is the document about?\"}'"
curl -s -X POST "$API_URL/ask-graph" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the document about?"}' | jq '.' || curl -s -X POST "$API_URL/ask-graph" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the document about?"}'
echo ""
echo ""

echo -e "${GREEN}[3.2] Ask-graph with custom thread_id${NC}"
echo "Command: curl -X POST $API_URL/ask-graph -H \"Content-Type: application/json\" -d '{\"question\": \"What are the requirements?\", \"thread_id\": \"$THREAD_ID\"}'"
curl -s -X POST "$API_URL/ask-graph" \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"What are the requirements?\", \"thread_id\": \"$THREAD_ID\"}" | jq '.' || curl -s -X POST "$API_URL/ask-graph" \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"What are the requirements?\", \"thread_id\": \"$THREAD_ID\"}"
echo ""
echo ""

if [ -n "$DOC_ID" ] && [ "$DOC_ID" != "null" ]; then
    echo -e "${GREEN}[3.3] Ask-graph with doc_id${NC}"
    echo "Command: curl -X POST $API_URL/ask-graph -H \"Content-Type: application/json\" -d '{\"question\": \"What are the main sections?\", \"thread_id\": \"$THREAD_ID\", \"doc_id\": \"$DOC_ID\"}'"
    curl -s -X POST "$API_URL/ask-graph" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"What are the main sections?\", \"thread_id\": \"$THREAD_ID\", \"doc_id\": \"$DOC_ID\"}" | jq '.' || curl -s -X POST "$API_URL/ask-graph" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"What are the main sections?\", \"thread_id\": \"$THREAD_ID\", \"doc_id\": \"$DOC_ID\"}"
    echo ""
    echo ""
    
    echo -e "${GREEN}[3.4] Ask-graph with doc_id and cross-doc${NC}"
    echo "Command: curl -X POST $API_URL/ask-graph -H \"Content-Type: application/json\" -d '{\"question\": \"What are the technical details?\", \"thread_id\": \"$THREAD_ID\", \"doc_id\": \"$DOC_ID\", \"cross_doc\": true}'"
    curl -s -X POST "$API_URL/ask-graph" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"What are the technical details?\", \"thread_id\": \"$THREAD_ID\", \"doc_id\": \"$DOC_ID\", \"cross_doc\": true}" | jq '.' || curl -s -X POST "$API_URL/ask-graph" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"What are the technical details?\", \"thread_id\": \"$THREAD_ID\", \"doc_id\": \"$DOC_ID\", \"cross_doc\": true}"
    echo ""
    echo ""
else
    echo -e "${YELLOW}[3.3-3.4] Skipping doc_id-specific queries (no doc_id available)${NC}"
    echo ""
fi

echo -e "${GREEN}[3.5] Ask-graph with cross-doc (no doc_id)${NC}"
echo "Command: curl -X POST $API_URL/ask-graph -H \"Content-Type: application/json\" -d '{\"question\": \"What are the key points?\", \"thread_id\": \"$THREAD_ID\", \"cross_doc\": true}'"
curl -s -X POST "$API_URL/ask-graph" \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"What are the key points?\", \"thread_id\": \"$THREAD_ID\", \"cross_doc\": true}" | jq '.' || curl -s -X POST "$API_URL/ask-graph" \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"What are the key points?\", \"thread_id\": \"$THREAD_ID\", \"cross_doc\": true}"
echo ""
echo ""

# =============================================================================
# 4. INFER ENDPOINTS (Direct Pipeline - Ingest + Query)
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}4. TESTING INFER ENDPOINTS (Direct Pipeline)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}[4.1] Infer with PDF (ingest + query)${NC}"
echo "Command: curl -X POST $API_URL/infer -F \"question=What does this document say?\" -F \"attachment=@$SAMPLE_PDF\" -F \"title=Inferred PDF\""
curl -s -X POST "$API_URL/infer" \
    -F "question=What does this document say?" \
    -F "attachment=@$SAMPLE_PDF" \
    -F "title=Inferred PDF" | jq '.' || curl -s -X POST "$API_URL/infer" \
    -F "question=What does this document say?" \
    -F "attachment=@$SAMPLE_PDF" \
    -F "title=Inferred PDF"
echo ""
echo ""

echo -e "${GREEN}[4.2] Infer with PDF and cross-doc${NC}"
echo "Command: curl -X POST $API_URL/infer -F \"question=What are the main points?\" -F \"attachment=@$SAMPLE_PDF\" -F \"title=Inferred PDF Cross-Doc\" -F \"cross_doc=true\""
curl -s -X POST "$API_URL/infer" \
    -F "question=What are the main points?" \
    -F "attachment=@$SAMPLE_PDF" \
    -F "title=Inferred PDF Cross-Doc" \
    -F "cross_doc=true" | jq '.' || curl -s -X POST "$API_URL/infer" \
    -F "question=What are the main points?" \
    -F "attachment=@$SAMPLE_PDF" \
    -F "title=Inferred PDF Cross-Doc" \
    -F "cross_doc=true"
echo ""
echo ""

echo -e "${GREEN}[4.3] Infer with Image${NC}"
echo "Command: curl -X POST $API_URL/infer -F \"question=What is shown in this image?\" -F \"attachment=@$SAMPLE_IMAGE2\" -F \"title=Inferred Image\""
curl -s -X POST "$API_URL/infer" \
    -F "question=What is shown in this image?" \
    -F "attachment=@$SAMPLE_IMAGE2" \
    -F "title=Inferred Image" | jq '.' || curl -s -X POST "$API_URL/infer" \
    -F "question=What is shown in this image?" \
    -F "attachment=@$SAMPLE_IMAGE2" \
    -F "title=Inferred Image"
echo ""
echo ""

echo -e "${GREEN}[4.4] Infer query-only (no attachment)${NC}"
echo "Command: curl -X POST $API_URL/infer -F \"question=What are the key points?\" -F \"cross_doc=false\""
curl -s -X POST "$API_URL/infer" \
    -F "question=What are the key points?" \
    -F "cross_doc=false" | jq '.' || curl -s -X POST "$API_URL/infer" \
    -F "question=What are the key points?" \
    -F "cross_doc=false"
echo ""
echo ""

# =============================================================================
# 5. INFER-GRAPH ENDPOINTS (LangGraph Pipeline - Ingest + Query)
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}5. TESTING INFER-GRAPH ENDPOINTS (LangGraph Pipeline)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}[5.1] Infer-graph with PDF${NC}"
echo "Command: curl -X POST $API_URL/infer-graph -F \"question=What does this document contain?\" -F \"attachment=@$SAMPLE_PDF\" -F \"title=Inferred Graph PDF\" -F \"thread_id=$THREAD_ID\""
curl -s -X POST "$API_URL/infer-graph" \
    -F "question=What does this document contain?" \
    -F "attachment=@$SAMPLE_PDF" \
    -F "title=Inferred Graph PDF" \
    -F "thread_id=$THREAD_ID" | jq '.' || curl -s -X POST "$API_URL/infer-graph" \
    -F "question=What does this document contain?" \
    -F "attachment=@$SAMPLE_PDF" \
    -F "title=Inferred Graph PDF" \
    -F "thread_id=$THREAD_ID"
echo ""
echo ""

echo -e "${GREEN}[5.2] Infer-graph with PDF and cross-doc${NC}"
echo "Command: curl -X POST $API_URL/infer-graph -F \"question=What are the technical requirements?\" -F \"attachment=@$SAMPLE_PDF\" -F \"title=Inferred Graph PDF Cross-Doc\" -F \"thread_id=$THREAD_ID\" -F \"cross_doc=true\""
curl -s -X POST "$API_URL/infer-graph" \
    -F "question=What are the technical requirements?" \
    -F "attachment=@$SAMPLE_PDF" \
    -F "title=Inferred Graph PDF Cross-Doc" \
    -F "thread_id=$THREAD_ID" \
    -F "cross_doc=true" | jq '.' || curl -s -X POST "$API_URL/infer-graph" \
    -F "question=What are the technical requirements?" \
    -F "attachment=@$SAMPLE_PDF" \
    -F "title=Inferred Graph PDF Cross-Doc" \
    -F "thread_id=$THREAD_ID" \
    -F "cross_doc=true"
echo ""
echo ""

echo -e "${GREEN}[5.3] Infer-graph with Image${NC}"
echo "Command: curl -X POST $API_URL/infer-graph -F \"question=What is in this image?\" -F \"attachment=@$SAMPLE_IMAGE\" -F \"title=Inferred Graph Image\" -F \"thread_id=$THREAD_ID\""
curl -s -X POST "$API_URL/infer-graph" \
    -F "question=What is in this image?" \
    -F "attachment=@$SAMPLE_IMAGE" \
    -F "title=Inferred Graph Image" \
    -F "thread_id=$THREAD_ID" | jq '.' || curl -s -X POST "$API_URL/infer-graph" \
    -F "question=What is in this image?" \
    -F "attachment=@$SAMPLE_IMAGE" \
    -F "title=Inferred Graph Image" \
    -F "thread_id=$THREAD_ID"
echo ""
echo ""

echo -e "${GREEN}[5.4] Infer-graph query-only (no attachment)${NC}"
echo "Command: curl -X POST $API_URL/infer-graph -F \"question=What are the key points?\" -F \"thread_id=$THREAD_ID\" -F \"cross_doc=false\""
curl -s -X POST "$API_URL/infer-graph" \
    -F "question=What are the key points?" \
    -F "thread_id=$THREAD_ID" \
    -F "cross_doc=false" | jq '.' || curl -s -X POST "$API_URL/infer-graph" \
    -F "question=What are the key points?" \
    -F "thread_id=$THREAD_ID" \
    -F "cross_doc=false"
echo ""
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TEST SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ All endpoint tests completed${NC}"
echo -e "${GREEN}✓ Check logs in: inference/graph/logs/${NC}"
echo -e "${GREEN}✓ Check thread_tracking table for audit logs${NC}"
echo -e "${GREEN}✓ API URL: $API_URL${NC}"
echo ""

