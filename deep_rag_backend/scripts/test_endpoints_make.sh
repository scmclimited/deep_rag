#!/bin/bash
# Comprehensive test script for all Make-based endpoints
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
echo -e "${BLUE}Deep RAG Endpoint Testing (Make Commands)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Store doc_id for subsequent queries
DOC_ID=""
THREAD_ID="test-thread-$(date +%s)"

# =============================================================================
# 1. INGEST ENDPOINTS
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}1. TESTING INGEST ENDPOINTS${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}[1.1] Ingest PDF (via Make)${NC}"
echo "Command: make cli-ingest FILE=\"$SAMPLE_PDF\" DOCKER=true TITLE=\"Test PDF Document\""
RESULT=$(make cli-ingest FILE="$SAMPLE_PDF" DOCKER=true TITLE="Test PDF Document" 2>&1)
echo "$RESULT"
DOC_ID=$(echo "$RESULT" | grep -oP 'doc_id:\s*\K[0-9a-f-]+' | head -1 || echo "")
if [ -n "$DOC_ID" ]; then
    echo -e "${GREEN}✓ Document ingested with doc_id: $DOC_ID${NC}"
else
    echo -e "${RED}✗ Failed to extract doc_id${NC}"
fi
echo ""

echo -e "${GREEN}[1.2] Ingest Image (via Make)${NC}"
echo "Command: make cli-ingest FILE=\"$SAMPLE_IMAGE\" DOCKER=true TITLE=\"Test Image Document\""
RESULT=$(make cli-ingest FILE="$SAMPLE_IMAGE" DOCKER=true TITLE="Test Image Document" 2>&1)
echo "$RESULT"
echo ""

# =============================================================================
# 2. QUERY ENDPOINTS (Direct Pipeline)
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}2. TESTING QUERY ENDPOINTS (Direct Pipeline)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}[2.1] Query all documents (no doc_id)${NC}"
echo "Command: make query Q=\"What is this document about?\" DOCKER=true"
make query Q="What is this document about?" DOCKER=true
echo ""

if [ -n "$DOC_ID" ]; then
    echo -e "${GREEN}[2.2] Query specific document (with doc_id)${NC}"
    echo "Command: make query Q=\"What are the main requirements?\" DOCKER=true DOC_ID=\"$DOC_ID\""
    make query Q="What are the main requirements?" DOCKER=true DOC_ID="$DOC_ID"
    echo ""
    
    echo -e "${GREEN}[2.3] Query with cross-doc enabled (with doc_id)${NC}"
    echo "Command: make query Q=\"What are the technical requirements?\" DOCKER=true DOC_ID=\"$DOC_ID\" CROSS_DOC=true"
    make query Q="What are the technical requirements?" DOCKER=true DOC_ID="$DOC_ID" CROSS_DOC=true
    echo ""
else
    echo -e "${YELLOW}[2.2-2.3] Skipping doc_id-specific queries (no doc_id available)${NC}"
    echo ""
fi

echo -e "${GREEN}[2.4] Query all documents with cross-doc${NC}"
echo "Command: make query Q=\"What are the key points?\" DOCKER=true CROSS_DOC=true"
make query Q="What are the key points?" DOCKER=true CROSS_DOC=true
echo ""

# =============================================================================
# 3. QUERY-GRAPH ENDPOINTS (LangGraph Pipeline)
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}3. TESTING QUERY-GRAPH ENDPOINTS (LangGraph Pipeline)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}[3.1] Query-graph all documents (default thread_id)${NC}"
echo "Command: make query-graph Q=\"What is the document about?\" DOCKER=true"
make query-graph Q="What is the document about?" DOCKER=true
echo ""

echo -e "${GREEN}[3.2] Query-graph with custom thread_id${NC}"
echo "Command: make query-graph Q=\"What are the requirements?\" DOCKER=true THREAD_ID=\"$THREAD_ID\""
make query-graph Q="What are the requirements?" DOCKER=true THREAD_ID="$THREAD_ID"
echo ""

if [ -n "$DOC_ID" ]; then
    echo -e "${GREEN}[3.3] Query-graph with doc_id${NC}"
    echo "Command: make query-graph Q=\"What are the main sections?\" DOCKER=true DOC_ID=\"$DOC_ID\" THREAD_ID=\"$THREAD_ID\""
    make query-graph Q="What are the main sections?" DOCKER=true DOC_ID="$DOC_ID" THREAD_ID="$THREAD_ID"
    echo ""
    
    echo -e "${GREEN}[3.4] Query-graph with doc_id and cross-doc${NC}"
    echo "Command: make query-graph Q=\"What are the technical details?\" DOCKER=true DOC_ID=\"$DOC_ID\" THREAD_ID=\"$THREAD_ID\" CROSS_DOC=true"
    make query-graph Q="What are the technical details?" DOCKER=true DOC_ID="$DOC_ID" THREAD_ID="$THREAD_ID" CROSS_DOC=true
    echo ""
else
    echo -e "${YELLOW}[3.3-3.4] Skipping doc_id-specific queries (no doc_id available)${NC}"
    echo ""
fi

echo -e "${GREEN}[3.5] Query-graph with cross-doc (no doc_id)${NC}"
echo "Command: make query-graph Q=\"What are the key points?\" DOCKER=true THREAD_ID=\"$THREAD_ID\" CROSS_DOC=true"
make query-graph Q="What are the key points?" DOCKER=true THREAD_ID="$THREAD_ID" CROSS_DOC=true
echo ""

# =============================================================================
# 4. INFER ENDPOINTS (Direct Pipeline - Ingest + Query)
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}4. TESTING INFER ENDPOINTS (Direct Pipeline)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}[4.1] Infer with PDF (ingest + query)${NC}"
echo "Command: make infer Q=\"What does this document say?\" FILE=\"$SAMPLE_PDF\" TITLE=\"Inferred PDF\" DOCKER=true"
make infer Q="What does this document say?" FILE="$SAMPLE_PDF" TITLE="Inferred PDF" DOCKER=true
echo ""

echo -e "${GREEN}[4.2] Infer with PDF and cross-doc${NC}"
echo "Command: make infer Q=\"What are the main points?\" FILE=\"$SAMPLE_PDF\" TITLE=\"Inferred PDF Cross-Doc\" DOCKER=true CROSS_DOC=true"
make infer Q="What are the main points?" FILE="$SAMPLE_PDF" TITLE="Inferred PDF Cross-Doc" DOCKER=true CROSS_DOC=true
echo ""

echo -e "${GREEN}[4.3] Infer with Image${NC}"
echo "Command: make infer Q=\"What is shown in this image?\" FILE=\"$SAMPLE_IMAGE2\" TITLE=\"Inferred Image\" DOCKER=true"
make infer Q="What is shown in this image?" FILE="$SAMPLE_IMAGE2" TITLE="Inferred Image" DOCKER=true
echo ""

# =============================================================================
# 5. INFER-GRAPH ENDPOINTS (LangGraph Pipeline - Ingest + Query)
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}5. TESTING INFER-GRAPH ENDPOINTS (LangGraph Pipeline)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}[5.1] Infer-graph with PDF${NC}"
echo "Command: make infer-graph Q=\"What does this document contain?\" FILE=\"$SAMPLE_PDF\" TITLE=\"Inferred Graph PDF\" DOCKER=true THREAD_ID=\"$THREAD_ID\""
make infer-graph Q="What does this document contain?" FILE="$SAMPLE_PDF" TITLE="Inferred Graph PDF" DOCKER=true THREAD_ID="$THREAD_ID"
echo ""

echo -e "${GREEN}[5.2] Infer-graph with PDF and cross-doc${NC}"
echo "Command: make infer-graph Q=\"What are the technical requirements?\" FILE=\"$SAMPLE_PDF\" TITLE=\"Inferred Graph PDF Cross-Doc\" DOCKER=true THREAD_ID=\"$THREAD_ID\" CROSS_DOC=true"
make infer-graph Q="What are the technical requirements?" FILE="$SAMPLE_PDF" TITLE="Inferred Graph PDF Cross-Doc" DOCKER=true THREAD_ID="$THREAD_ID" CROSS_DOC=true
echo ""

echo -e "${GREEN}[5.3] Infer-graph with Image${NC}"
echo "Command: make infer-graph Q=\"What is in this image?\" FILE=\"$SAMPLE_IMAGE\" TITLE=\"Inferred Graph Image\" DOCKER=true THREAD_ID=\"$THREAD_ID\""
make infer-graph Q="What is in this image?" FILE="$SAMPLE_IMAGE" TITLE="Inferred Graph Image" DOCKER=true THREAD_ID="$THREAD_ID"
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
echo ""

