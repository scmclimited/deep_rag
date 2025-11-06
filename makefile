# Tools
PY ?= python
DC ?= docker compose

# Defaults (override like: make ingest FILE=./samples/doc.pdf)
FILE ?=
Q ?=
OUT ?= deep_rag_graph.png

.PHONY: help up down logs rebuild db-up db-down ingest query query-graph infer-graph graph inspect

help:
	@echo ""
	@echo "Deep RAG — common commands"
	@echo "---------------------------"
	@echo "make up            # build & start API + DB via root docker-compose.yml"
	@echo "make down          # stop and remove containers/volumes"
	@echo "make logs          # tail API + DB logs"
	@echo "make rebuild       # rebuild API image and restart stack"
	@echo "make db-up         # start DB only (vector_db/docker-compose.yml)"
	@echo "make db-down       # stop DB-only stack"
	@echo "make ingest FILE=path/to/file.pdf"
	@echo "make cli-ingest FILE=path/to/file.pdf [DOCKER=true] [TITLE='Document Title']"
	@echo "make query  Q='your question here'  # uses agent_loop.py (direct pipeline)"
	@echo "make query-graph  Q='your question here'  # uses LangGraph (with conditional routing)"
	@echo "make infer-graph  Q='your question' [FILE=path/to/file.pdf] [TITLE='Title']  # uses LangGraph"
	@echo "make graph  OUT=deep_rag_graph.png [DOCKER=true]  # export LangGraph diagram (PNG or Mermaid fallback)"
	@echo "make inspect [TITLE='Document Title'] [DOC_ID=uuid] [DOCKER=true]  # inspect stored chunks and pages for a document"
	@echo ""

# --- Root stack (API + DB) ---
up:
	$(DC) up -d --build

down:
	$(DC) down -v

logs:
	$(DC) logs -f

rebuild:
	$(DC) build --no-cache
	$(DC) up -d

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
		docker compose exec api python -m ingestion.ingest "$(FILE)"; \
	else \
		$(PY) ingestion/ingest.py "$(FILE)"; \
	fi

query:
	@if [ -z "$(Q)" ]; then echo "Usage: make query Q='your question' [DOCKER=true]"; exit 2; fi
	@if [ "$(DOCKER)" = "true" ]; then \
		docker compose exec api python -m inference.cli query "$(Q)"; \
	else \
		$(PY) inference/cli.py query "$(Q)"; \
	fi

query-graph:
	@if [ -z "$(Q)" ]; then echo "Usage: make query-graph Q='your question' [DOCKER=true] [THREAD_ID=default]"; exit 2; fi
	@if [ "$(DOCKER)" = "true" ]; then \
		if [ -n "$(THREAD_ID)" ]; then \
			docker compose exec api python -m inference.cli query-graph "$(Q)" --thread-id "$(THREAD_ID)"; \
		else \
			docker compose exec api python -m inference.cli query-graph "$(Q)"; \
		fi \
	else \
		if [ -n "$(THREAD_ID)" ]; then \
			$(PY) inference/cli.py query-graph "$(Q)" --thread-id "$(THREAD_ID)"; \
		else \
			$(PY) inference/cli.py query-graph "$(Q)"; \
		fi \
	fi

cli-ingest:
	@if [ -z "$(FILE)" ]; then echo "Usage: make cli-ingest FILE=path/to/file.pdf [DOCKER=true] [TITLE='Document Title']"; exit 2; fi
	@if [ "$(DOCKER)" = "true" ]; then \
		if [ -n "$(TITLE)" ]; then \
			docker compose exec api python -m inference.cli ingest "$(FILE)" --title "$(TITLE)"; \
		else \
			docker compose exec api python -m inference.cli ingest "$(FILE)"; \
		fi \
	else \
		if [ -n "$(TITLE)" ]; then \
			$(PY) inference/cli.py ingest "$(FILE)" --title "$(TITLE)"; \
		else \
			$(PY) inference/cli.py ingest "$(FILE)"; \
		fi \
	fi

# --- LangGraph inference (ingest + query with graph) ---
infer-graph:
	@if [ -z "$(Q)" ]; then echo "Usage: make infer-graph Q='your question' [FILE=path/to/file.pdf] [TITLE='Title'] [DOCKER=true] [THREAD_ID=default]"; exit 2; fi
	@if [ "$(DOCKER)" = "true" ]; then \
		if [ -n "$(FILE)" ]; then \
			if [ -n "$(TITLE)" ]; then \
				if [ -n "$(THREAD_ID)" ]; then \
					docker compose exec api python -m inference.cli infer-graph "$(Q)" --file "$(FILE)" --title "$(TITLE)" --thread-id "$(THREAD_ID)"; \
				else \
					docker compose exec api python -m inference.cli infer-graph "$(Q)" --file "$(FILE)" --title "$(TITLE)"; \
				fi \
			else \
				if [ -n "$(THREAD_ID)" ]; then \
					docker compose exec api python -m inference.cli infer-graph "$(Q)" --file "$(FILE)" --thread-id "$(THREAD_ID)"; \
				else \
					docker compose exec api python -m inference.cli infer-graph "$(Q)" --file "$(FILE)"; \
				fi \
			fi \
		else \
			if [ -n "$(THREAD_ID)" ]; then \
				docker compose exec api python -m inference.cli infer-graph "$(Q)" --thread-id "$(THREAD_ID)"; \
			else \
				docker compose exec api python -m inference.cli infer-graph "$(Q)"; \
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

# --- Graph visualization ---
# Use DOCKER=true to run inside Docker container where dependencies are installed
graph:
	@if [ "$(DOCKER)" = "true" ]; then \
		docker compose exec api python -m inference.graph.graph_viz --out "$(OUT)"; \
	else \
		$(PY) inference/graph/graph_viz.py --out "$(OUT)"; \
	fi
	@echo "Graph written to $(OUT) (or .mmd fallback if Graphviz is missing)"

# --- Diagnostics ---
# Inspect what chunks and pages are stored for a document
# Use TITLE for document title search (partial match) or DOC_ID for exact document ID
inspect:
	@if [ "$(DOCKER)" = "true" ]; then \
		if [ -n "$(DOC_ID)" ]; then \
			docker compose exec api python -m inference.cli inspect --doc-id "$(DOC_ID)"; \
		elif [ -n "$(TITLE)" ]; then \
			docker compose exec api python -m inference.cli inspect --title "$(TITLE)"; \
		else \
			docker compose exec api python -m inference.cli inspect; \
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
