# Vector DB Service Makefile
# Logging commands for database service

# Tools
DC ?= docker compose

# Defaults
TAIL ?= 500
FOLLOW ?= false
OUTPUT ?=

.PHONY: help logs logs-db logs-follow

help:
	@echo ""
	@echo "Vector DB Service - Logging Commands"
	@echo "===================================="
	@echo "  make logs [TAIL=500] [OUTPUT=db_logs.txt] [FOLLOW=false]  # Capture DB logs to file"
	@echo "  make logs-db [TAIL=500] [OUTPUT=db_logs.txt] [FOLLOW=false]  # Capture DB logs"
	@echo "  make logs-follow [TAIL=500]  # Follow DB logs in real-time"
	@echo ""

# Dynamic logs command - capture service logs to file
logs:
	@if [ -z "$(OUTPUT)" ]; then \
		OUTPUT_FILE="db_logs.txt"; \
	else \
		OUTPUT_FILE="$(OUTPUT)"; \
	fi; \
	if [ "$(FOLLOW)" = "true" ]; then \
		echo "Following logs from service 'db' (last $(TAIL) lines) to $$OUTPUT_FILE..."; \
		$(DC) logs --tail=$(TAIL) -f db | tee $$OUTPUT_FILE; \
	else \
		echo "Capturing logs from service 'db' (last $(TAIL) lines) to $$OUTPUT_FILE..."; \
		$(DC) logs --tail=$(TAIL) db > $$OUTPUT_FILE 2>&1; \
		echo "✓ Logs saved to $$OUTPUT_FILE"; \
	fi

# Convenience shortcut for DB service
logs-db:
	@$(MAKE) logs TAIL=$(TAIL) OUTPUT=$(OUTPUT) FOLLOW=$(FOLLOW)

# Follow logs in real-time
logs-follow:
	$(DC) logs -f --tail=$(TAIL) db

