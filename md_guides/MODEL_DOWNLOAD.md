# Pre-downloading Models for Docker

This guide explains how to pre-download the CLIP and reranker models so they don't need to be downloaded every time you rebuild the Docker image.

## Prerequisites

### Python 3.11+ Required

Ensure you have Python 3.11 or higher installed:

```bash
python --version  # Should show 3.11 or higher
```

### Activate Python Environment

Before running the download script, you must activate your Python environment:

#### Option 1: Python Virtual Environment (venv)

```bash
# From project root
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Verify activation (should show (.venv) in prompt)
which python  # Linux/macOS
where python  # Windows
```

#### Option 2: Conda Environment

**Note**: Conda must be installed separately. If you get `conda: command not found`, either:
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- Or use **Option 1 (venv)** instead, which is built into Python and requires no additional installation

If conda is installed:

```bash
# Create conda environment with Python 3.11
conda create -n deep_rag python=3.11

# Activate conda environment
conda activate deep_rag

# Verify activation (should show (deep_rag) in prompt)
conda info --envs
```

### Install Required Dependencies

After activating your environment, install the minimal dependencies needed for downloading models:

```bash
cd deep_rag_backend

# Install minimal dependencies for model download
pip install -r requirements-download.txt

# Or install manually:
pip install transformers sentence-transformers torch
```

**Note**: If you encounter issues installing PyMuPDF or other dependencies, you can skip them for now since they're not needed for downloading models. See `deep_rag_backend/WINDOWS_SETUP.md` for Windows-specific troubleshooting.

## Why Pre-download Models?

- **Faster Docker rebuilds**: Models are baked into the image, no download needed at runtime
- **Offline capability**: Works without internet connection after initial download
- **Consistent versions**: Ensures the same model version is used every time
- **Faster startup**: No model download delay when containers start

## Method 1: Pre-download and Bake into Docker Image (Recommended)

### Step 1: Download the Models Locally

**Make sure your Python environment is activated** (see Prerequisites above), then run:

```bash
# Ensure your Python environment is activated first!
# (venv: source .venv/bin/activate or .venv\Scripts\activate)
# (conda: conda activate deep_rag)

cd deep_rag_backend
python scripts/download_model.py
```

This will download both:
- **CLIP model** (~3.4 GB) to `./models/openai_clip-vit-large-patch14-336/`
- **Reranker model** (~100-200 MB) to `./models/cross-encoder_ms-marco-MiniLM-L-6-v2/`

Or download models individually:

```bash
# Download only CLIP model
python scripts/download_model.py --clip-only

# Download only reranker model
python scripts/download_model.py --reranker-only

# Download with custom models
python scripts/download_model.py --model openai/clip-vit-large-patch14-336 --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Step 2: Build Docker Image

The Dockerfile will automatically copy the `models/` directory into the image:

```bash
# From project root
docker compose build api
```

The model files will be baked into the Docker image, so no download is needed at runtime.

### Step 3: Use the Models

The model loading code will automatically detect the local model paths. You can also explicitly set them in your `.env`:

```bash
# Optional: Explicitly set the model paths
CLIP_MODEL_PATH=/app/models/openai_clip-vit-large-patch14-336
CLIP_MODEL=openai/clip-vit-large-patch14-336  # Still needed for fallback
EMBEDDING_DIM=768

RERANK_MODEL_PATH=/app/models/cross-encoder_ms-marco-MiniLM-L-6-v2
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  # Still needed for fallback
```

## Method 2: Use Docker Volume Mount (Alternative)

If you prefer to keep models outside the image:

### Step 1: Download Model Locally

**Make sure your Python environment is activated** (see Prerequisites above), then run:

```bash
# Ensure your Python environment is activated first!
# (venv: source .venv/bin/activate or .venv\Scripts\activate)
# (conda: conda activate deep_rag)

cd deep_rag_backend
python scripts/download_model.py
```

### Step 2: Mount Models as Volume

Update `docker-compose.yml`:

```yaml
services:
  api:
    volumes:
      - ./deep_rag_backend/models:/app/models:ro  # Read-only mount
    environment:
      CLIP_MODEL_PATH: /app/models/openai_clip-vit-large-patch14-336
      RERANK_MODEL_PATH: /app/models/cross-encoder_ms-marco-MiniLM-L-6-v2
```

### Step 3: Start Services

```bash
docker compose up
```

## Method 3: Use Hugging Face Cache (Runtime Download)

If you want models downloaded at runtime (slower first startup, but no pre-download needed):

1. **Remove models from .dockerignore** (or don't create models directory)
2. **Let transformers download automatically** on first run
3. **Use Docker volume for cache persistence**:

```yaml
services:
  api:
    volumes:
      - huggingface_cache:/root/.cache/huggingface
volumes:
  huggingface_cache:
```

## Model Loading Priority

The model loading code checks in this order:

1. `CLIP_MODEL_PATH` / `RERANK_MODEL_PATH` environment variable (if set and path exists)
2. `CLIP_MODEL` / `RERANK_MODEL` environment variable (downloads from Hugging Face)
3. Falls back to Hugging Face if local path doesn't exist

## File Structure

After downloading, your models directory will look like:

```
deep_rag_backend/
  models/
    openai_clip-vit-large-patch14-336/
      config.json
      pytorch_model.bin (or model.safetensors)
      preprocessor_config.json
      tokenizer.json
      tokenizer_config.json
      vocab.json
      merges.txt
      special_tokens_map.json
      ...
    cross-encoder_ms-marco-MiniLM-L-6-v2/
      config.json
      pytorch_model.bin
      tokenizer files...
      ...
```

## Model Sizes

- **openai/clip-vit-large-patch14-336**: ~3.4 GB
- **cross-encoder/ms-marco-MiniLM-L-6-v2**: ~100-200 MB
- **Total**: ~3.6 GB

## Troubleshooting

### ModuleNotFoundError: transformers

**Error**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**: 
1. Make sure your Python environment is activated (see Prerequisites)
2. Install dependencies:
   ```bash
   pip install -r requirements-download.txt
   # Or: pip install transformers sentence-transformers torch
   ```

### Python Environment Not Activated

**Error**: Script runs but uses system Python instead of virtual environment

**Solution**:
- **venv**: Run `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
- **conda**: Run `conda activate deep_rag` (only if conda is installed)
- Verify with `which python` (Linux/macOS) or `where python` (Windows) - should point to your environment

### Conda Command Not Found

**Error**: `bash: conda: command not found` or `conda: command not found`

**Solution**: 
- **Option 1 (Recommended)**: Use Python's built-in venv instead (no installation needed):
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Linux/macOS
  # or
  .venv\Scripts\activate  # Windows
  ```

- **Option 2**: Install conda:
  - Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight)
  - Or install [Anaconda](https://www.anaconda.com/download) (full distribution)
  - After installation, restart your terminal and try again

- **Option 3**: Add conda to PATH (if conda is installed but not in PATH)

#### Adding Conda to PATH on Windows

If conda is installed but not recognized, you need to add it to your PATH:

**Method 1: Using Anaconda Prompt (Easiest)**
1. Open "Anaconda Prompt" or "Anaconda PowerShell Prompt" from Start Menu
2. Conda is automatically available in these terminals
3. Run your commands from there

**Method 2: Add to System PATH (Permanent)**

1. **Find conda installation path** (usually one of these):
   - `C:\Users\<YourUsername>\anaconda3` (user installation)
   - `C:\Users\<YourUsername>\miniconda3` (user installation)
   - `C:\ProgramData\anaconda3` (system-wide installation - **your case**)
   - `C:\ProgramData\miniconda3` (system-wide installation)

2. **Add to PATH via System Settings**:
   - Press `Win + X` and select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "User variables" or "System variables", find and select "Path", then click "Edit"
   - Click "New" and add these paths (adjust based on your installation):
     - `C:\ProgramData\anaconda3` (or `C:\Users\<YourUsername>\anaconda3` for user install)
     - `C:\ProgramData\anaconda3\Scripts` (or `C:\Users\<YourUsername>\anaconda3\Scripts`)
     - `C:\ProgramData\anaconda3\Library\bin` (or `C:\Users\<YourUsername>\anaconda3\Library\bin`)
   - Click "OK" on all dialogs
   - **Restart your terminal** (close and reopen)

3. **Verify conda is in PATH**:
   ```bash
   # In PowerShell or CMD
   where conda
   
   # In Git Bash
   which conda
   ```

**Method 3: Initialize Conda for Git Bash**

If using Git Bash, you need to use Unix-style paths (forward slashes, `/c/` instead of `C:\`):

```bash
# For Anaconda in ProgramData (common installation location):
/c/ProgramData/anaconda3/Scripts/conda.exe init bash

# For Anaconda in user directory:
/c/Users/<YourUsername>/anaconda3/Scripts/conda.exe init bash

# For Miniconda:
/c/Users/<YourUsername>/miniconda3/Scripts/conda.exe init bash

# Restart Git Bash after running the init command
```

**Important**: In Git Bash, always use forward slashes (`/`) and `/c/` instead of `C:\`. The path `C:\ProgramData\anaconda3` becomes `/c/ProgramData/anaconda3` in Git Bash.

**Method 4: Temporary PATH (Current Session Only)**

For a quick fix in your current terminal session:

```bash
# PowerShell
$env:Path += ";C:\ProgramData\anaconda3;C:\ProgramData\anaconda3\Scripts;C:\ProgramData\anaconda3\Library\bin"

# Git Bash (use Unix-style paths with /c/ instead of C:\)
export PATH="/c/ProgramData/anaconda3:/c/ProgramData/anaconda3/Scripts:/c/ProgramData/anaconda3/Library/bin:$PATH"

# CMD
set PATH=%PATH%;C:\ProgramData\anaconda3;C:\ProgramData\anaconda3\Scripts;C:\ProgramData\anaconda3\Library\bin
```

**For Git Bash with ProgramData installation** (your case):
```bash
# Add conda to PATH for current session
export PATH="/c/ProgramData/anaconda3:/c/ProgramData/anaconda3/Scripts:/c/ProgramData/anaconda3/Library/bin:$PATH"

# Verify it works
conda --version
```

**Note**: If conda is installed in a different location, adjust the paths accordingly:
- User directory: `/c/Users/<YourUsername>/anaconda3` (Git Bash) or `C:\Users\<YourUsername>\anaconda3` (PowerShell/CMD)
- ProgramData: `/c/ProgramData/anaconda3` (Git Bash) or `C:\ProgramData\anaconda3` (PowerShell/CMD)

**Verify Installation**:
```bash
conda --version
conda info
```

### Wrong Python Version

**Error**: Python version issues or compatibility errors

**Solution**:
- Ensure Python 3.11+ is installed: `python --version`
- Create new environment with correct version:
  ```bash
  # venv
  python3.11 -m venv .venv
  
  # conda
  conda create -n deep_rag python=3.11
  ```

### Model Not Found

If you get "Model not found" errors:

1. Check that the model path exists: `ls -la deep_rag_backend/models/`
2. Verify `CLIP_MODEL_PATH` is set correctly in `.env`
3. Check Docker logs: `docker compose logs api`

### Docker Build Fails

If Docker build fails when copying models:

1. Make sure models directory exists: `ls deep_rag_backend/models/`
2. Check `.dockerignore` doesn't exclude `models/`
3. Verify model files are complete (re-download if needed)

### Permission Issues

If you get permission errors:

```bash
# Fix permissions
chmod -R 755 deep_rag_backend/models/
```

## References

- [Hugging Face Model Hub](https://huggingface.co/openai/clip-vit-large-patch14-336)
- [Transformers Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/clip)

