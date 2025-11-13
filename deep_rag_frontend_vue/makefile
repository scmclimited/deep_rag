# Frontend Service Makefile
# Logging commands for frontend service

# Tools
DC ?= docker compose

# Defaults
TAIL ?= 500
FOLLOW ?= false
OUTPUT ?=

.PHONY: help logs logs-frontend logs-follow

help:
	@echo ""
	@echo "Frontend Service - Logging Commands"
	@echo "===================================="
	@echo "  make logs [TAIL=500] [OUTPUT=frontend_logs.txt] [FOLLOW=false]  # Capture frontend logs to file"
	@echo "  make logs-frontend [TAIL=500] [OUTPUT=frontend_logs.txt] [FOLLOW=false]  # Capture frontend logs"
	@echo "  make logs-follow [TAIL=500]  # Follow frontend logs in real-time"
	@echo ""

# Dynamic logs command - capture service logs to file
logs:
	@if [ -z "$(OUTPUT)" ]; then \
		OUTPUT_FILE="frontend_logs.txt"; \
	else \
		OUTPUT_FILE="$(OUTPUT)"; \
	fi; \
	if [ "$(FOLLOW)" = "true" ]; then \
		echo "Following logs from service 'frontend' (last $(TAIL) lines) to $$OUTPUT_FILE..."; \
		cd .. && $(DC) logs --tail=$(TAIL) -f frontend | tee deep_rag_frontend_vue/$$OUTPUT_FILE; \
	else \
		echo "Capturing logs from service 'frontend' (last $(TAIL) lines) to $$OUTPUT_FILE..."; \
		cd .. && $(DC) logs --tail=$(TAIL) frontend > deep_rag_frontend_vue/$$OUTPUT_FILE 2>&1; \
		echo "✓ Logs saved to $$OUTPUT_FILE"; \
	fi

# Convenience shortcut for frontend service
logs-frontend:
	@$(MAKE) logs TAIL=$(TAIL) OUTPUT=$(OUTPUT) FOLLOW=$(FOLLOW)

# Follow logs in real-time
logs-follow:
	cd .. && $(DC) logs -f --tail=$(TAIL) frontend

