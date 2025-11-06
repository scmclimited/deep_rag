# LLM Provider Setup Guide

## Quick Fix: Switch to Ollama, Gemini, or OpenAI for Text-Only Chat

**Problem**: LLaVA is a vision-language model and may not work well for text-only chat tasks. The "ModelWrapper" error suggests compatibility issues.

**Solution**: Use Ollama, Gemini, or OpenAI for text-only RAG tasks.

### Option 1: Use Ollama (Recommended for Local Development)

1. **Install Ollama** (if not already installed):
   ```bash
   # Windows: Download from https://ollama.ai/download
   # macOS: brew install ollama
   # Linux: curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull a model**:
   ```bash
   ollama pull llama3:8b
   # Or use a smaller model:
   ollama pull llama3.2:3b
   ```

3. **Start Ollama**:
   ```bash
   ollama serve
   ```

4. **Update your `.env` file**:
   ```bash
   LLM_PROVIDER=ollama
   OLLAMA_URL=http://localhost:11434
   OLLAMA_MODEL=llama3:8b
   ```

5. **Restart Docker**:
   ```bash
   docker compose restart api
   ```

### Option 2: Use Gemini (Multi-Modal, Cloud-Based) ⭐ Recommended

Gemini is multi-modal (text, images, audio, video) and excellent for RAG tasks.

1. **Get an API key** from https://makersuite.google.com/app/apikey

2. **Update your `.env` file**:
   ```bash
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-pro
   # Or use the latest: gemini-1.5-pro
   ```

3. **Restart Docker**:
   ```bash
   docker compose restart api
   ```

**Benefits of Gemini**:
- ✅ Multi-modal (can handle text, images, audio, video)
- ✅ Excellent for RAG tasks
- ✅ Competitive pricing
- ✅ Strong reasoning capabilities

### Option 3: Use OpenAI (Cloud-Based)

1. **Get an API key** from https://platform.openai.com/api-keys

2. **Update your `.env` file**:
   ```bash
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```

3. **Restart Docker**:
   ```bash
   docker compose restart api
   ```

### Option 4: Fix LLaVA (If You Must Use It)

If you need to use LLaVA for vision tasks:

1. **Clear corrupted model cache**:
   ```bash
   # Inside Docker container
   docker compose exec api bash
   rm -rf ~/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf
   ```

2. **Re-download the model**:
   ```bash
   # The model will re-download on next use
   ```

3. **Note**: LLaVA is designed for vision-language tasks. For text-only RAG, Ollama or OpenAI will work much better.

## Testing the LLM

After switching providers, test with:

```bash
docker compose exec api python -m inference.cli query "What is the technical assessment about?"
```

## Provider Comparison

| Provider | Best For | Multi-Modal | Setup | Cost |
|----------|----------|-------------|-------|------|
| **Gemini** | Production, RAG, multi-modal tasks | ✅ Yes | Easy (API key) | Free tier available, then paid |
| **Ollama** | Local text-only chat | ❌ No | Easy (install + pull model) | Free |
| **OpenAI** | Production, cloud-based | ✅ Yes (GPT-4o) | Easy (API key) | Paid |
| **LLaVA** | Vision-language tasks (local) | ✅ Yes | Complex (large model) | Free but slow |

## Troubleshooting

### Ollama Connection Error
- Ensure `ollama serve` is running
- Check `OLLAMA_URL` in `.env` matches your Ollama service
- Verify the model is pulled: `ollama list`

### Gemini API Error
- Verify `GEMINI_API_KEY` is set correctly
- Check API key is valid at https://makersuite.google.com/app/apikey
- Ensure `GEMINI_MODEL` is valid (e.g., "gemini-pro" or "gemini-1.5-pro")
- Check API quota/limits

### OpenAI API Error
- Verify `OPENAI_API_KEY` is set correctly
- Check API key has sufficient credits
- Ensure `OPENAI_MODEL` is valid

### LLaVA Errors
- Model may be corrupted - try clearing cache
- LLaVA may not work well for text-only - switch to Ollama/OpenAI
- Ensure sufficient GPU memory (or it will be very slow on CPU)

