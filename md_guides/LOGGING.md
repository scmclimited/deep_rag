# Dynamic Logging Commands

This guide explains how to use the dynamic logging commands to capture Docker Compose service logs to files.

## Overview

The logging system allows you to:
- Capture logs from any service (api, db, frontend)
- Specify the number of lines to capture (tail)
- Output to custom file names
- Follow logs in real-time

## Root Project Commands

### Makefile Commands

From the project root (`deep_rag/`):

```bash
# Basic usage - capture last 500 lines from api service to api_logs.txt
make logs SERVICE=api

# Custom tail amount
make logs SERVICE=api TAIL=1000

# Custom output file
make logs SERVICE=api OUTPUT=custom_api.log

# Follow logs in real-time
make logs SERVICE=api FOLLOW=true

# Convenience shortcuts
make logs-api                    # Capture API logs (default: 500 lines)
make logs-api TAIL=1000         # Capture last 1000 lines
make logs-api OUTPUT=api.log     # Custom output file
make logs-api FOLLOW=true        # Follow in real-time

make logs-db                     # Capture DB logs
make logs-frontend               # Capture frontend logs

# Follow all services
make logs-follow TAIL=500
```

### Python Script Commands

From the project root:

```bash
# Basic usage
python scripts/logs.py api

# Custom tail amount
python scripts/logs.py api --tail 1000

# Custom output file
python scripts/logs.py api --output custom_api.log

# Follow logs in real-time
python scripts/logs.py api --follow

# After installing package
logs api
logs api --tail 1000 --output api.log
logs api --follow
```

## Service-Specific Commands

### Backend Service (`deep_rag_backend/`)

From the backend directory:

```bash
# Makefile commands
make logs SERVICE=api TAIL=500 OUTPUT=api_logs.txt
make logs-api TAIL=1000

# Python script
python scripts/logs.py api --tail 500 --output api_logs.txt
```

### Vector DB Service (`vector_db/`)

From the vector_db directory:

```bash
# Makefile commands
make logs TAIL=500 OUTPUT=db_logs.txt
make logs-db TAIL=1000

# Python script
python scripts/logs.py --tail 500 --output db_logs.txt
```

### Frontend Service (`deep_rag_frontend_vue/`)

From the frontend directory:

```bash
# Makefile commands
make logs TAIL=500 OUTPUT=frontend_logs.txt
make logs-frontend TAIL=1000
```

## Examples

### Capture API logs for debugging

```bash
# From project root
make logs-api TAIL=1000 OUTPUT=api_debug.log

# Or using Python
python scripts/logs.py api --tail 1000 --output api_debug.log
```

### Monitor database in real-time

```bash
# From project root
make logs-db FOLLOW=true

# Or using Python
python scripts/logs.py db --follow
```

### Capture all service logs

```bash
# Capture each service separately
make logs-api TAIL=500 OUTPUT=api_logs.txt
make logs-db TAIL=500 OUTPUT=db_logs.txt
make logs-frontend TAIL=500 OUTPUT=frontend_logs.txt
```

## File Output

By default, logs are saved to:
- `{service}_logs.txt` (e.g., `api_logs.txt`, `db_logs.txt`, `frontend_logs.txt`)

You can override this with the `OUTPUT` parameter (Makefile) or `--output` flag (Python script).

## Notes

- When using `FOLLOW=true` or `--follow`, logs are written to the file in real-time
- Press `Ctrl+C` to stop following logs
- Logs include both stdout and stderr (combined)
- The scripts automatically find the correct `docker-compose.yml` file based on the directory structure

