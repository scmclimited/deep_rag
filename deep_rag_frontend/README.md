# Deep RAG Frontend

Streamlit-based frontend for Deep RAG, providing a ChatGPT-like interface for document ingestion and querying.

## Features

- 💬 **Chat Interface**: ChatGPT-like conversation interface
- 🧵 **Thread Management**: Create, switch, and manage conversation threads
- 📤 **File Upload**: Single and batch file ingestion (PDF, TXT, PNG, JPEG)
- 📚 **Document Management**: View and manage ingested documents
- 🔍 **Advanced Querying**: Support for cross-document search and document filtering
- 🎯 **LangGraph Integration**: Uses LangGraph pipeline for agentic reasoning

## Setup

### Prerequisites

- Python 3.11+ (required for Google Gemini SDK compatibility)
- Deep RAG backend running (default: http://localhost:8000)
- Docker and Docker Compose (for containerized deployment)

### Installation

#### Option 1: Docker (Recommended)

1. **Create `.env` file**:
```bash
cp .env.example .env
```

Edit `.env` and set `API_BASE_URL`:
- If backend is running locally: `API_BASE_URL=http://host.docker.internal:8000`
- If backend is in same Docker network: `API_BASE_URL=http://api:8000`
- If backend is remote: `API_BASE_URL=http://your-backend-url:8000`

2. **Build and run with Docker Compose**:
```bash
docker-compose up -d
```

The app will be available at `http://localhost:8501`

3. **View logs**:
```bash
docker-compose logs -f frontend
```

4. **Stop the frontend**:
```bash
docker-compose down
```

#### Option 2: Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables** (optional):
```bash
export API_BASE_URL=http://localhost:8000
```

Or create a `.env` file:
```
API_BASE_URL=http://localhost:8000
```

3. **Run the Streamlit app**:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

#### Option 3: Docker (Standalone)

1. **Build the image**:
```bash
docker build -t deep-rag-frontend .
```

2. **Run the container**:
```bash
docker run -d \
  --name deep-rag-frontend \
  -p 8501:8501 \
  -e API_BASE_URL=http://host.docker.internal:8000 \
  deep-rag-frontend
```

## Usage

### Chat Interface

1. **Ask Questions**: Type your question in the chat input and press Enter
2. **Attach Files**: Upload files to include with your question (files are ingested automatically)
3. **Filter Documents**: Use the sidebar to filter queries to specific documents
4. **Cross-Document Search**: Enable cross-document retrieval in settings

### File Ingestion

**Single File Upload:**
- Use the sidebar file uploader to ingest files without querying
- Or attach a file to your next message to ingest and query simultaneously

**Batch Upload:**
- Select multiple files in the sidebar uploader
- Click "Ingest Files" to process all files

### Thread Management

- **New Thread**: Click "➕ New Thread" to start a fresh conversation
- **Thread History**: (Coming soon) View and switch between previous threads
- Each thread maintains its own conversation context

### Document Management

- **View Documents**: Documents are listed in the sidebar
- **Refresh**: Click "🔄 Refresh Documents" to update the list
- **Delete**: (Coming soon) Delete documents from the knowledge base

## API Integration

The frontend uses the following Deep RAG API endpoints:

### Existing Endpoints Used:
- `GET /health` - Health check
- `POST /ingest` - Single file ingestion
- `POST /ask-graph` - Query with LangGraph pipeline
- `POST /infer-graph` - Ingest + query with LangGraph pipeline
- `GET /diagnostics/document` - Document diagnostics (fallback for document list)

### Suggested New Endpoints:

See `SUGGESTED_ROUTES.md` for recommended additional endpoints to enhance the experience.

## Configuration

### Environment Variables

- `API_BASE_URL`: Backend API URL (default: `http://localhost:8000`)

### Streamlit Configuration

Create `.streamlit/config.toml` for custom configuration:

```toml
[server]
port = 8501
enableCORS = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

## Docker Configuration

### Environment Variables

- `API_BASE_URL`: Backend API URL
  - Local backend: `http://host.docker.internal:8000`
  - Same network: `http://api:8000`
  - Remote: `http://your-backend-url:8000`
- `FRONTEND_PORT`: Frontend port (default: 8501)

### Docker Compose Options

**Standalone Frontend** (default):
```bash
docker-compose up -d
```

**Connect to Backend Network**:
If backend is running in a separate docker-compose, uncomment the network configuration in `docker-compose.yml`:
```yaml
networks:
  default:
    external:
      name: deep_rag_network
```

**Development Mode** (hot reload):
Uncomment the volume mount in `docker-compose.yml`:
```yaml
volumes:
  - .:/app
```

## Architecture

```
deep_rag_frontend/
├── app.py              # Main Streamlit application
├── api_client.py       # API client wrapper
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Docker Compose configuration
├── .dockerignore       # Docker ignore file
└── README.md          # This file
```

## Future Enhancements

- [ ] WebSocket/SSE support for streaming responses
- [ ] Thread history persistence and retrieval
- [ ] Document deletion
- [ ] User authentication
- [ ] Response streaming
- [ ] Export conversation history
- [ ] Advanced search filters
- [ ] Document preview
- [ ] Batch ingestion progress tracking

## Troubleshooting

### API Connection Issues

If you see "API is offline":
1. Ensure the Deep RAG backend is running
2. Check `API_BASE_URL` is correct
3. Verify backend is accessible from the frontend

### File Upload Issues

- Ensure file types are supported (PDF, TXT, PNG, JPEG)
- Check file size limits
- Verify backend has sufficient resources

### Thread Management

- Thread IDs are generated client-side
- Thread history requires backend endpoints (see `SUGGESTED_ROUTES.md`)

