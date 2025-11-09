# Tools
PY ?= python
DC ?= docker compose

# Defaults (override like: make ingest FILE=./samples/doc.pdf)
FILE ?=
Q ?=
OUT ?= deep_rag_graph.png

.PHONY: help up down logs rebuild db-up db-down ingest query query-graph infer-graph graph health inspect

help:
	@echo ""
	@echo "Deep RAG — common commands"
	@echo "---------------------------"
	@echo "make up            # build & start API + DB via root docker-compose.yml"
	@echo "make up-and-test   # build & start API + DB, then run tests to verify setup (optionally runs endpoint tests if AUTOMATE_ENDPOINT_RUNS_ON_BOOT=true)"
	@echo "make down          # stop and remove containers/volumes"
	@echo "make logs          # tail API + DB logs"
	@echo "make rebuild       # rebuild API image and restart stack"
	@echo "make db-up         # start DB only (vector_db/docker-compose.yml)"
	@echo "make db-down       # stop DB-only stack"
	@echo "make ingest FILE=path/to/file.pdf"
	@echo "make cli-ingest FILE=path/to/file.pdf [DOCKER=true] [TITLE='Document Title']"
	@echo "make query  Q='your question here'  # uses inference/agents/pipeline.py (direct pipeline)"
	@echo "make query-graph  Q='your question here'  # uses LangGraph (with conditional routing)"
	@echo "make infer  Q='your question' [FILE=path/to/file.pdf] [TITLE='Title']  # uses direct pipeline"
	@echo "make infer-graph  Q='your question' [FILE=path/to/file.pdf] [TITLE='Title']  # uses LangGraph"
	@echo "make graph  OUT=deep_rag_graph.png [DOCKER=true]  # export LangGraph diagram (PNG or Mermaid fallback)"
	@echo "make health  # check API and database health status"
	@echo "make inspect [TITLE='Document Title'] [DOC_ID=uuid] [DOCKER=true]  # inspect stored chunks and pages for a document"
	@echo "make test [DOCKER=true]  # run all tests (unit + integration)"
	@echo "make unit-tests [DOCKER=true]  # run unit tests only"
	@echo "make integration-tests [DOCKER=true]  # run integration tests only"
	@echo "make test-endpoints  # test all ingest/query/infer endpoints (Make + REST)"
	@echo "make test-endpoints-make  # test endpoints via Make commands"
	@echo "make test-endpoints-rest  # test endpoints via REST API (curl)"
	@echo "make test-endpoints-quick  # quick test (one example of each endpoint type)"
	@echo ""

# --- Root stack (API + DB) ---
up:
	$(DC) -f ../docker-compose.yml up -d --build
	@echo ""
	@echo "Services started. Waiting for API to be ready..."
	@sleep 3
	@echo "Run 'make test DOCKER=true' to verify everything is working."

up-and-test: up
	@echo ""
	@echo "Running tests to verify setup..."
	@$(MAKE) test DOCKER=true
	@if [ "${AUTOMATE_ENDPOINT_RUNS_ON_BOOT:-false}" = "true" ]; then \
		echo ""; \
		echo "=========================================="; \
		echo "Running endpoint tests (AUTOMATE_ENDPOINT_RUNS_ON_BOOT=true)"; \
		echo "=========================================="; \
		$(MAKE) test-endpoints; \
		echo "=========================================="; \
		echo "Endpoint tests completed."; \
		echo "=========================================="; \
	fi

down:
	$(DC) -f ../docker-compose.yml down -v

logs:
	$(DC) -f ../docker-compose.yml logs -f

rebuild:
	$(DC) -f ../docker-compose.yml build --no-cache
	$(DC) -f ../docker-compose.yml up -d

# --- DB-only stack (useful for local dev) ---
db-up:
	cd vector_db && docker-compose up -d

db-down:
	cd vector_db && docker-compose down -v

# --- CLI wrappers (run locally or in Docker) ---
# Use DOCKER=true to run inside Docker container
# Example: make ingest FILE=./file.pdf DOCKER=true
DOCKER ?= false

ingest:
	@if [ -z "$(FILE)" ]; then echo "Usage: make ingest FILE=path/to/file.pdf [DOCKER=true]"; exit 2; fi
	@if [ "$(DOCKER)" = "true" ]; then \
		FILE_PATH=$$(echo "$(FILE)" | tr '\\\\' '/'); \
		docker compose -f ../docker-compose.yml exec api python -m ingestion.ingest "$$FILE_PATH"; \
	else \
		$(PY) ingestion/ingest.py "$(FILE)"; \
	fi

query:
	@if [ -z "$(Q)" ]; then echo "Usage: make query Q='your question' [DOCKER=true] [DOC_ID=uuid] [CROSS_DOC=true]"; exit 2; fi
	@if [ "$(DOCKER)" = "true" ]; then \
		if [ -n "$(DOC_ID)" ]; then \
			if [ "$(CROSS_DOC)" = "true" ]; then \
				docker compose -f ../docker-compose.yml exec api python -m inference.cli query "$(Q)" --doc-id "$(DOC_ID)" --cross-doc; \
			else \
				docker compose -f ../docker-compose.yml exec api python -m inference.cli query "$(Q)" --doc-id "$(DOC_ID)"; \
			fi \
		else \
			if [ "$(CROSS_DOC)" = "true" ]; then \
				docker compose -f ../docker-compose.yml exec api python -m inference.cli query "$(Q)" --cross-doc; \
			else \
				docker compose -f ../docker-compose.yml exec api python -m inference.cli query "$(Q)"; \
			fi \
		fi \
	else \
		if [ -n "$(DOC_ID)" ]; then \
			if [ "$(CROSS_DOC)" = "true" ]; then \
				$(PY) inference/cli.py query "$(Q)" --doc-id "$(DOC_ID)" --cross-doc; \
			else \
				$(PY) inference/cli.py query "$(Q)" --doc-id "$(DOC_ID)"; \
			fi \
		else \
			if [ "$(CROSS_DOC)" = "true" ]; then \
				$(PY) inference/cli.py query "$(Q)" --cross-doc; \
			else \
				$(PY) inference/cli.py query "$(Q)"; \
			fi \
		fi \
	fi

query-graph:
	@if [ -z "$(Q)" ]; then echo "Usage: make query-graph Q='your question' [DOCKER=true] [THREAD_ID=default] [DOC_ID=uuid] [CROSS_DOC=true]"; exit 2; fi
	@if [ "$(DOCKER)" = "true" ]; then \
		if [ -n "$(DOC_ID)" ]; then \
			if [ -n "$(THREAD_ID)" ]; then \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli query-graph "$(Q)" --thread-id "$(THREAD_ID)" --doc-id "$(DOC_ID)" --cross-doc; \
				else \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli query-graph "$(Q)" --thread-id "$(THREAD_ID)" --doc-id "$(DOC_ID)"; \
				fi \
			else \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli query-graph "$(Q)" --doc-id "$(DOC_ID)" --cross-doc; \
				else \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli query-graph "$(Q)" --doc-id "$(DOC_ID)"; \
				fi \
			fi \
		else \
			if [ -n "$(THREAD_ID)" ]; then \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli query-graph "$(Q)" --thread-id "$(THREAD_ID)" --cross-doc; \
				else \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli query-graph "$(Q)" --thread-id "$(THREAD_ID)"; \
				fi \
			else \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli query-graph "$(Q)" --cross-doc; \
				else \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli query-graph "$(Q)"; \
				fi \
			fi \
		fi \
	else \
		if [ -n "$(DOC_ID)" ]; then \
			if [ -n "$(THREAD_ID)" ]; then \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					$(PY) inference/cli.py query-graph "$(Q)" --thread-id "$(THREAD_ID)" --doc-id "$(DOC_ID)" --cross-doc; \
				else \
					$(PY) inference/cli.py query-graph "$(Q)" --thread-id "$(THREAD_ID)" --doc-id "$(DOC_ID)"; \
				fi \
			else \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					$(PY) inference/cli.py query-graph "$(Q)" --doc-id "$(DOC_ID)" --cross-doc; \
				else \
					$(PY) inference/cli.py query-graph "$(Q)" --doc-id "$(DOC_ID)"; \
				fi \
			fi \
		else \
			if [ -n "$(THREAD_ID)" ]; then \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					$(PY) inference/cli.py query-graph "$(Q)" --thread-id "$(THREAD_ID)" --cross-doc; \
				else \
					$(PY) inference/cli.py query-graph "$(Q)" --thread-id "$(THREAD_ID)"; \
				fi \
			else \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					$(PY) inference/cli.py query-graph "$(Q)" --cross-doc; \
				else \
					$(PY) inference/cli.py query-graph "$(Q)"; \
				fi \
			fi \
		fi \
	fi

cli-ingest:
	@if [ -z "$(FILE)" ]; then echo "Usage: make cli-ingest FILE=path/to/file.pdf [DOCKER=true] [TITLE='Document Title']"; exit 2; fi
	@if [ "$(DOCKER)" = "true" ]; then \
		FILE_PATH=$$(echo "$(FILE)" | tr '\\\\' '/'); \
		if [ -n "$(TITLE)" ]; then \
			docker compose -f ../docker-compose.yml exec api python -m inference.cli ingest "$$FILE_PATH" --title "$(TITLE)"; \
		else \
			docker compose -f ../docker-compose.yml exec api python -m inference.cli ingest "$$FILE_PATH"; \
		fi \
	else \
		if [ -n "$(TITLE)" ]; then \
			$(PY) inference/cli.py ingest "$(FILE)" --title "$(TITLE)"; \
		else \
			$(PY) inference/cli.py ingest "$(FILE)"; \
		fi \
	fi

# --- Direct inference (ingest + query without graph) ---
infer:
	@if [ -z "$(Q)" ]; then echo "Usage: make infer Q='your question' [FILE=path/to/file.pdf] [TITLE='Title'] [DOCKER=true] [CROSS_DOC=true]"; exit 2; fi
	@if [ "$(DOCKER)" = "true" ]; then \
		if [ -n "$(FILE)" ]; then \
			FILE_PATH=$$(echo "$(FILE)" | tr '\\\\' '/'); \
			if [ -n "$(TITLE)" ]; then \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli infer "$(Q)" --file "$$FILE_PATH" --title "$(TITLE)" --cross-doc; \
				else \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli infer "$(Q)" --file "$$FILE_PATH" --title "$(TITLE)"; \
				fi \
			else \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli infer "$(Q)" --file "$$FILE_PATH" --cross-doc; \
				else \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli infer "$(Q)" --file "$$FILE_PATH"; \
				fi \
			fi \
		else \
			if [ "$(CROSS_DOC)" = "true" ]; then \
				docker compose -f ../docker-compose.yml exec api python -m inference.cli infer "$(Q)" --cross-doc; \
			else \
				docker compose -f ../docker-compose.yml exec api python -m inference.cli infer "$(Q)"; \
			fi \
		fi \
	else \
		if [ -n "$(FILE)" ]; then \
			if [ -n "$(TITLE)" ]; then \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					$(PY) inference/cli.py infer "$(Q)" --file "$(FILE)" --title "$(TITLE)" --cross-doc; \
				else \
					$(PY) inference/cli.py infer "$(Q)" --file "$(FILE)" --title "$(TITLE)"; \
				fi \
			else \
				if [ "$(CROSS_DOC)" = "true" ]; then \
					$(PY) inference/cli.py infer "$(Q)" --file "$(FILE)" --cross-doc; \
				else \
					$(PY) inference/cli.py infer "$(Q)" --file "$(FILE)"; \
				fi \
			fi \
		else \
			if [ "$(CROSS_DOC)" = "true" ]; then \
				$(PY) inference/cli.py infer "$(Q)" --cross-doc; \
			else \
				$(PY) inference/cli.py infer "$(Q)"; \
			fi \
		fi \
	fi

# --- LangGraph inference (ingest + query with graph) ---
infer-graph:
	@if [ -z "$(Q)" ]; then echo "Usage: make infer-graph Q='your question' [FILE=path/to/file.pdf] [TITLE='Title'] [DOCKER=true] [THREAD_ID=default]"; exit 2; fi
	@if [ "$(DOCKER)" = "true" ]; then \
		if [ -n "$(FILE)" ]; then \
			FILE_PATH=$$(echo "$(FILE)" | tr '\\\\' '/'); \
			if [ -n "$(TITLE)" ]; then \
				if [ -n "$(THREAD_ID)" ]; then \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli infer-graph "$(Q)" --file "$$FILE_PATH" --title "$(TITLE)" --thread-id "$(THREAD_ID)"; \
				else \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli infer-graph "$(Q)" --file "$$FILE_PATH" --title "$(TITLE)"; \
				fi \
			else \
				if [ -n "$(THREAD_ID)" ]; then \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli infer-graph "$(Q)" --file "$$FILE_PATH" --thread-id "$(THREAD_ID)"; \
				else \
					docker compose -f ../docker-compose.yml exec api python -m inference.cli infer-graph "$(Q)" --file "$$FILE_PATH"; \
				fi \
			fi \
		else \
			if [ -n "$(THREAD_ID)" ]; then \
				docker compose -f ../docker-compose.yml exec api python -m inference.cli infer-graph "$(Q)" --thread-id "$(THREAD_ID)"; \
			else \
				docker compose -f ../docker-compose.yml exec api python -m inference.cli infer-graph "$(Q)"; \
			fi \
		fi \
	else \
		if [ -n "$(FILE)" ]; then \
			if [ -n "$(TITLE)" ]; then \
				if [ -n "$(THREAD_ID)" ]; then \
					$(PY) inference/cli.py infer-graph "$(Q)" --file "$(FILE)" --title "$(TITLE)" --thread-id "$(THREAD_ID)"; \
				else \
					$(PY) inference/cli.py infer-graph "$(Q)" --file "$(FILE)" --title "$(TITLE)"; \
				fi \
			else \
				if [ -n "$(THREAD_ID)" ]; then \
					$(PY) inference/cli.py infer-graph "$(Q)" --file "$(FILE)" --thread-id "$(THREAD_ID)"; \
				else \
					$(PY) inference/cli.py infer-graph "$(Q)" --file "$(FILE)"; \
				fi \
			fi \
		else \
			if [ -n "$(THREAD_ID)" ]; then \
				$(PY) inference/cli.py infer-graph "$(Q)" --thread-id "$(THREAD_ID)"; \
			else \
				$(PY) inference/cli.py infer-graph "$(Q)"; \
			fi \
		fi \
	fi

# --- Testing ---
test:
	@echo "Running all tests..."
	@$(MAKE) unit-tests DOCKER=$(DOCKER)
	@$(MAKE) integration-tests DOCKER=$(DOCKER)

unit-tests:
	@echo "Running unit tests..."
	@if [ "$(DOCKER)" = "true" ]; then \
		docker compose -f ../docker-compose.yml exec api python -m pytest tests/unit/ -v; \
	else \
		$(PY) -m pytest tests/unit/ -v; \
	fi

integration-tests:
	@echo "Running integration tests..."
	@if [ "$(DOCKER)" = "true" ]; then \
		docker compose -f ../docker-compose.yml exec api python -m pytest tests/integration/ -v; \
	else \
		$(PY) -m pytest tests/integration/ -v; \
	fi

# --- Endpoint Testing ---
test-endpoints-make:
	@echo "Testing all endpoints via Make commands..."
	@bash scripts/test_endpoints_make.sh

test-endpoints-rest:
	@echo "Testing all endpoints via REST API..."
	@bash scripts/test_endpoints_rest.sh

test-endpoints: test-endpoints-make test-endpoints-rest
	@echo "All endpoint tests completed!"

test-endpoints-quick:
	@echo "Running quick endpoint test (one example of each type)..."
	@bash scripts/test_endpoints_quick.sh

# --- Graph visualization ---
# Use DOCKER=true to run inside Docker container where dependencies are installed
graph:
	@if [ "$(DOCKER)" = "true" ]; then \
		docker compose -f ../docker-compose.yml exec api python -m inference.graph.graph_viz --out "$(OUT)"; \
	else \
		$(PY) inference/graph/graph_viz.py --out "$(OUT)"; \
	fi
	@echo "Graph written to $(OUT) (or .mmd fallback if Graphviz is missing)"

# --- Health Check ---
health:
	@echo "Checking API health..."
	@curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || curl -s http://localhost:8000/health || echo "Error: API is not running. Please run 'make up' first."

# --- Diagnostics ---
# Inspect what chunks and pages are stored for a document
# Use TITLE for document title search (partial match) or DOC_ID for exact document ID
inspect:
	@if [ "$(DOCKER)" = "true" ]; then \
		if [ -n "$(DOC_ID)" ]; then \
			docker compose -f ../docker-compose.yml exec api python -m inference.cli inspect --doc-id "$(DOC_ID)"; \
		elif [ -n "$(TITLE)" ]; then \
			docker compose -f ../docker-compose.yml exec api python -m inference.cli inspect --title "$(TITLE)"; \
		else \
			docker compose -f ../docker-compose.yml exec api python -m inference.cli inspect; \
		fi \
	else \
		if [ -n "$(DOC_ID)" ]; then \
			$(PY) inference/cli.py inspect --doc-id "$(DOC_ID)"; \
		elif [ -n "$(TITLE)" ]; then \
			$(PY) inference/cli.py inspect --title "$(TITLE)"; \
		else \
			$(PY) inference/cli.py inspect; \
		fi \
	fi
