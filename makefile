# Deep RAG - Root Project Makefile
# This Makefile orchestrates all services (database, backend API, frontend)
# and delegates to component-specific Makefiles when needed.

# Tools
PY ?= python
DC ?= docker compose

# Defaults
FILE ?=
Q ?=
OUT ?= deep_rag_backend/inference/graph/artifacts/deep_rag_graph.png
DOCKER ?= false

# Component directories
BACKEND_DIR := deep_rag_backend
FRONTEND_DIR := deep_rag_frontend
VECTOR_DB_DIR := vector_db

.PHONY: help up down logs rebuild up-and-test
.PHONY: db-up db-down
.PHONY: backend frontend
.PHONY: cli-ingest query query-graph infer infer-graph graph health inspect
.PHONY: test unit-tests integration-tests test-endpoints test-endpoints-make test-endpoints-rest test-endpoints-quick
.PHONY: clean-cache

help:
	@echo ""
	@echo "Deep RAG - Root Project Commands"
	@echo "================================="
	@echo ""
	@echo "Full Stack Orchestration:"
	@echo "  make up              # Start all services (db + api + frontend)"
	@echo "  make up-and-test     # Start all services, then run tests"
	@echo "  make down            # Stop all services (keeps volumes - data persists)"
	@echo "  make down-clean      # Stop all services and remove volumes (deletes data)"
	@echo "  make logs [SERVICE=api] [TAIL=500] [OUTPUT=api_logs.txt] [FOLLOW=false]  # Capture service logs to file"
	@echo "  make logs-api [TAIL=500] [OUTPUT=api_logs.txt] [FOLLOW=false]  # Capture API logs"
	@echo "  make logs-db [TAIL=500] [OUTPUT=db_logs.txt] [FOLLOW=false]  # Capture DB logs"
	@echo "  make logs-frontend [TAIL=500] [OUTPUT=frontend_logs.txt] [FOLLOW=false]  # Capture frontend logs"
	@echo "  make logs-follow [TAIL=500]  # Follow all services logs in real-time"
	@echo "  make rebuild          # Rebuild all images and restart stack"
	@echo ""
	@echo "Database Only:"
	@echo "  make db-up            # Start database only"
	@echo "  make db-down          # Stop database only"
	@echo ""
	@echo "Backend Commands (delegated to deep_rag_backend/makefile):"
	@echo "  make cli-ingest FILE=path/to/file.pdf [DOCKER=true] [TITLE='Title']"
	@echo "  make query Q='question' [DOCKER=true] [DOC_ID=uuid] [CROSS_DOC=true]"
	@echo "  make query-graph Q='question' [DOCKER=true] [THREAD_ID=id] [DOC_ID=uuid] [CROSS_DOC=true]"
	@echo "  make infer Q='question' [FILE=path] [TITLE='Title'] [DOCKER=true] [CROSS_DOC=true]"
	@echo "  make infer-graph Q='question' [FILE=path] [TITLE='Title'] [DOCKER=true] [THREAD_ID=id]"
	@echo "  make graph OUT=deep_rag_backend/inference/graph/artifacts/deep_rag_graph.png [DOCKER=true]"
	@echo "  make health"
	@echo "  make inspect [TITLE='Title'] [DOC_ID=uuid] [DOCKER=true]"
	@echo ""
	@echo "Testing:"
	@echo "  make test [DOCKER=true]              # Run all tests"
	@echo "  make unit-tests [DOCKER=true]         # Run unit tests only"
	@echo "  make integration-tests [DOCKER=true]  # Run integration tests only"
	@echo "  make test-endpoints                   # Test all endpoints (Make + REST)"
	@echo "  make test-endpoints-make              # Test endpoints via Make commands"
	@echo "  make test-endpoints-rest              # Test endpoints via REST API"
	@echo "  make test-endpoints-quick             # Quick endpoint test"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean-cache                      # Remove all Python cache files (__pycache__, .pyc, .pyo)"
	@echo ""
	@echo "Note: All commands can be run from the project root (deep_rag/)."
	@echo "      Backend-specific commands are delegated to deep_rag_backend/makefile"
	@echo "      with proper directory context."
	@echo ""

# --- Full Stack Orchestration ---
up:
	@echo "Starting all services (database, backend API, frontend)..."
	$(DC) up -d --build
	@echo ""
	@echo "Services started. Waiting for services to be ready..."
	@sleep 5
	@echo "Services available at:"
	@echo "  - Frontend: http://localhost:5173"
	@echo "  - Backend API: http://localhost:8000"
	@echo "  - Database: localhost:5432"
	@echo ""
	@echo "Run 'make test DOCKER=true' to verify everything is working."

up-and-test: up
	@echo ""
	@echo "Running tests to verify setup..."
	@$(MAKE) -C $(BACKEND_DIR) test DOCKER=true
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
	@echo "Stopping all services (keeping volumes - data will persist)..."
	$(DC) down
	@echo "All services stopped. Data is preserved in named volume 'deep_rag_db_data'."
	@echo "To remove data and start fresh, use: make down-clean"

down-clean:
	@echo "Stopping all services and removing volumes (data will be deleted)..."
	$(DC) down -v
	@echo "All services stopped and volumes removed."

# Dynamic logs command - capture service logs to file
# Usage: make logs SERVICE=api TAIL=500 OUTPUT=api_logs.txt
# Usage: make logs SERVICE=api TAIL=500 FOLLOW=true
SERVICE ?= api
TAIL ?= 500
FOLLOW ?= false
OUTPUT ?=

logs:
	@if [ -z "$(OUTPUT)" ]; then \
		OUTPUT_FILE="$(SERVICE)_logs.txt"; \
	else \
		OUTPUT_FILE="$(OUTPUT)"; \
	fi; \
	if [ "$(FOLLOW)" = "true" ]; then \
		echo "Following logs from service '$(SERVICE)' (last $(TAIL) lines) to $$OUTPUT_FILE..."; \
		$(DC) logs --tail=$(TAIL) -f $(SERVICE) | tee $$OUTPUT_FILE; \
	else \
		echo "Capturing logs from service '$(SERVICE)' (last $(TAIL) lines) to $$OUTPUT_FILE..."; \
		$(DC) logs --tail=$(TAIL) $(SERVICE) > $$OUTPUT_FILE 2>&1; \
		echo "Logs saved to $$OUTPUT_FILE"; \
	fi

# Convenience shortcuts for each service
logs-api:
	@$(MAKE) logs SERVICE=api TAIL=$(TAIL) OUTPUT=$(OUTPUT) FOLLOW=$(FOLLOW)

logs-db:
	@$(MAKE) logs SERVICE=db TAIL=$(TAIL) OUTPUT=$(OUTPUT) FOLLOW=$(FOLLOW)

logs-frontend:
	@$(MAKE) logs SERVICE=frontend TAIL=$(TAIL) OUTPUT=$(OUTPUT) FOLLOW=$(FOLLOW)

# Follow logs in real-time (all services)
logs-follow:
	$(DC) logs -f --tail=$(TAIL)

rebuild:
	@echo "Rebuilding all images..."
	$(DC) build --no-cache
	$(DC) up -d
	@echo "All services rebuilt and restarted."

# --- Database Only ---
db-up:
	@echo "Starting database only..."
	@cd $(VECTOR_DB_DIR) && docker-compose up -d
	@echo "Database started at localhost:5432"

db-down:
	@echo "Stopping database only..."
	@cd $(VECTOR_DB_DIR) && docker-compose down -v
	@echo "Database stopped."

# --- Backend Commands (delegated with proper directory context) ---
# These commands delegate to the backend Makefile, ensuring proper directory context

cli-ingest:
	@$(MAKE) -C $(BACKEND_DIR) cli-ingest FILE="$(FILE)" DOCKER="$(DOCKER)" TITLE="$(TITLE)"

query:
	@$(MAKE) -C $(BACKEND_DIR) query Q="$(Q)" DOCKER="$(DOCKER)" DOC_ID="$(DOC_ID)" CROSS_DOC="$(CROSS_DOC)"

query-graph:
	@$(MAKE) -C $(BACKEND_DIR) query-graph Q="$(Q)" DOCKER="$(DOCKER)" THREAD_ID="$(THREAD_ID)" DOC_ID="$(DOC_ID)" CROSS_DOC="$(CROSS_DOC)"

infer:
	@$(MAKE) -C $(BACKEND_DIR) infer Q="$(Q)" FILE="$(FILE)" TITLE="$(TITLE)" DOCKER="$(DOCKER)" CROSS_DOC="$(CROSS_DOC)"

infer-graph:
	@$(MAKE) -C $(BACKEND_DIR) infer-graph Q="$(Q)" FILE="$(FILE)" TITLE="$(TITLE)" DOCKER="$(DOCKER)" THREAD_ID="$(THREAD_ID)"

graph:
	@$(MAKE) -C $(BACKEND_DIR) graph OUT="$(OUT)" DOCKER="$(DOCKER)"

health:
	@$(MAKE) -C $(BACKEND_DIR) health

inspect:
	@$(MAKE) -C $(BACKEND_DIR) inspect TITLE="$(TITLE)" DOC_ID="$(DOC_ID)" DOCKER="$(DOCKER)"

# --- Testing (delegated to backend) ---
test:
	@$(MAKE) -C $(BACKEND_DIR) test DOCKER="$(DOCKER)"

unit-tests:
	@$(MAKE) -C $(BACKEND_DIR) unit-tests DOCKER="$(DOCKER)"

integration-tests:
	@$(MAKE) -C $(BACKEND_DIR) integration-tests DOCKER="$(DOCKER)"

test-endpoints:
	@$(MAKE) -C $(BACKEND_DIR) test-endpoints

test-endpoints-make:
	@$(MAKE) -C $(BACKEND_DIR) test-endpoints-make

test-endpoints-rest:
	@$(MAKE) -C $(BACKEND_DIR) test-endpoints-rest

test-endpoints-quick:
	@$(MAKE) -C $(BACKEND_DIR) test-endpoints-quick

# --- Utilities ---
clean-cache:
	@$(MAKE) -C $(BACKEND_DIR) clean-cache

