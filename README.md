# LunaCore — FastAPI + Ollama + Chroma RAG

A local, privacy‑friendly assistant that uses:
- **FastAPI** for the HTTP API
- **Ollama** for running LLMs locally
- **ChromaDB** for vector search (long‑term memory + RAG)
- Optional **Discord**-friendly streaming endpoint

This README covers **installation, configuration, and quickstart** with sample requests.

---

## 1) Prerequisites

- **Python** 3.11+
- **Git** (optional)
- **Ollama** installed and running locally  
  ↳ https://ollama.com (install and ensure `ollama` CLI works)
- **Models to pull** with Ollama:
  ```bash
  # One general LLM (choose one you like; examples):
  ollama pull llama3.1:8b
  # or: ollama pull qwen2.5:7b   # example alternative
  # Embedding model:
  ollama pull nomic-embed-text
  ```
- **Windows users**: install IANA timezone data (fixes ZoneInfo errors)
  ```bash
  pip install tzdata
  ```

---

## 2) Create & activate a virtualenv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

---

## 3) Install dependencies

If your repo has a `requirements.txt`:
```bash
pip install -U pip
pip install -r requirements.txt
```

If not, the minimum set often looks like:
```bash
pip install -U pip
pip install fastapi uvicorn chromadb ollama pydantic tzdata
```

> **Note:** `tzdata` is harmless on macOS/Linux and helpful on Windows.

---

## 4) Environment variables

Create a `.env` (optional) or set environment variables before launching. Common ones:

```bash
# Ollama settings
export OLLAMA_HOST="http://127.0.0.1:11434"   # default if local

# Luna generation defaults (used by app.services.ollama_service)
export LUNA_KEEP_ALIVE="30m"
export LUNA_NUM_CTX="4096"
export LUNA_NUM_PREDICT="256"
export LUNA_TEMPERATURE="0.2"
export LUNA_TOP_K="30"
export LUNA_TOP_P="0.9"

# Timezone default (used for time-aware answers)
export LUNA_DEFAULT_TZ="Asia/Ho_Chi_Minh"

# Chroma persistence (optional)
export CHROMA_PATH="chromadb_store"
```

On Windows PowerShell use `setx` or `$env:VAR="value"`.

---

## 5) Start Ollama

In one terminal:
```bash
ollama serve
```
Make sure your models are pulled (see step 1).

---

## 6) Run the API

In another terminal (inside the project root):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
> If your entrypoint differs, adjust `app.main:app` accordingly (e.g. `main:app`).

Basic health check:
```bash
curl http://localhost:8000/health
```

---

## 7) Key Endpoints (overview)

- `GET  /health` — simple service check
- `GET  /stats` — Chroma collection stats
- `GET  /cache/embeddings` — embedding cache counters
- `POST /cache/embeddings/clear` — clear the embedding cache
- `POST /ingest` — add documents to the vector store
- `POST /delete` — delete by ids/filters/content
- `GET  /search` — quick relevance search (debug)
- `POST /chat` — regular chat (JSON request/response)
- `POST /chat/stream` — Server‑Sent Events stream
- `POST /discord/chat/stream` — SSE stream tuned for Discord
- `GET  /memory/get|summary|list` — long‑term memory utilities

> Authentication is not included by default. Protect these endpoints if you expose them publicly.

---

## 8) Ingest sample data

Example: store a public org overview document and tag it for RAG.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "Neon Moon Studio is an indie game studio of dreamers crafting emotionally stirring stories in luminous, alternate realities. We focus on narrative-first experiences, cozy mechanics, and striking neon aesthetics."
    ],
    "metadatas": {
      "path": "/org/neon_moon/overview.md",
      "filename": "overview.md",
      "category": "org",
      "tags": ["neon-moon","overview","process","stack"],
      "visibility": "public",
      "source": "api"
    },
    "chunk": true,
    "chunk_size": 800,
    "overlap": 120
  }'
```

You can also ingest **multiple** docs with per-doc metadata using the `items` shape:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "text": "Team: Nova (Creative Director), Orion (Lead Engineer), Selene (Narrative Designer).",
        "metadata": {
          "path": "/org/neon_moon/team.md",
          "filename": "team.md",
          "category": "org",
          "tags": ["neon-moon","team"],
          "visibility": "public",
          "source": "api"
        }
      },
      {
        "text": "Project Starfall: A cozy narrative adventure about constellations and memory.",
        "metadata": {
          "path": "/org/neon_moon/projects/starfall.md",
          "filename": "starfall.md",
          "category": "project",
          "tags": ["neon-moon","starfall","design"],
          "visibility": "public",
          "source": "api"
        }
      }
    ],
    "chunk": true,
    "chunk_size": 800,
    "overlap": 120
  }'
```

Check stats:
```bash
curl http://localhost:8000/stats
```

---

## 9) Chat examples

### 9.1 Synchronous chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "user_id": "alice",
    "user_name": "Alice",
    "conversation_id": "session-1",
    "user_tz": "Asia/Ho_Chi_Minh",
    "messages": [
      {"role": "user", "content": "Give me a one-paragraph overview of Neon Moon Studio."}
    ],
    "use_server_memory": false
  }'
```

### 9.2 Streaming chat (SSE)
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "user_id": "alice",
    "user_name": "Alice",
    "conversation_id": "session-2",
    "user_tz": "Asia/Ho_Chi_Minh",
    "messages": [
      {"role": "user", "content": "List the team and current project in bullet points."}
    ],
    "use_server_memory": false
  }'
```

### 9.3 Discord-flavored streaming (no citation rule)
```bash
curl -N -X POST http://localhost:8000/discord/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "user_id": "alice",
    "user_name": "Alice",
    "conversation_id": "discord-channel-42",
    "user_tz": "Asia/Ho_Chi_Minh",
    "messages": [
      {"role": "user", "content": "What is Project Starfall about?"}
    ]
  }'
```

---

## 10) Deleting data

### 10.1 Delete by metadata filter
```bash
curl -X POST http://localhost:8000/delete \
  -H "Content-Type: application/json" \
  -d '{
    "where": {"path": "/org/neon_moon/team.md"}
  }'
```

### 10.2 Delete by content (document contains substring)
```bash
curl -X POST http://localhost:8000/delete \
  -H "Content-Type: application/json" \
  -d '{
    "where_document": {"$contains": "constellations and memory"}
  }'
```

### 10.3 Drop the entire collection (dangerous)
```bash
curl -X POST "http://localhost:8000/collections/drop?name=luna-memory"
```

---

## 11) Performance tips

- Keep **Ollama** running (`ollama serve`) and consider `keep_alive=30m` to avoid cold starts.
- Pull quantized models if CPU‑only; keep **context** reasonable (`LUNA_NUM_CTX=4096` is a good start).
- Use the embedding cache endpoints to monitor savings; warm up by hitting `/search` once after boot.
- On Windows, prefer **WSL2** for better CPU performance if available.
- If you see slow first‑response times, it may be model load time or first‑query Chroma index warmup.

---

## 12) Troubleshooting

- **`ZoneInfoNotFoundError` on Windows** → `pip install tzdata` and/or use valid IANA names like `Etc/UTC` or `Asia/Ho_Chi_Minh`.
- **`model "string" not found`** → you passed `"string"` as the model; set a real tag like `llama3.1:8b` and ensure it’s pulled.
- **Chroma “include ids” error** → This project already avoids `include=["ids"]` in `query/get`. If you fork older code, remove `ids` from `include`.
- **Very empty/irrelevant RAG** → Ingest more content, use realistic chunking (`chunk_size` 600–1000, `overlap` 80–150), and ensure metadata `visibility":"public"` (or pass `user_id` and allow_public).

---

## 13) Notes on chunking

- `chunk` enables splitting large documents into overlapping blocks—improves recall for long texts.
- `chunk_size` is the target block size in characters. Try **800–1200** for general prose.
- `overlap` keeps context continuity between chunks (80–150 is common).

---

## 14) Security & deployment

These endpoints ship **without auth** by default. For production:
- Put the service behind a reverse proxy (nginx, Caddy) with HTTPS.
- Add an auth layer (API keys, JWT, or your identity provider).
- Keep your Ollama host private or protected; it has powerful local access.

---

## 15) Project layout (typical)

```
LunaCore/
  app/
    main.py              # FastAPI app creation + routers
    router.py            # endpoints (chat/ingest/etc)
    services/ollama_service.py
    rag/embedder.py
    rag/retriever.py
    memory/auto.py
    memory/store.py
    memory/summarizer.py
  chromadb_store/        # persisted vectors (default path)
  luna_orgPesona.txt     # persona system prompt
  requirements.txt
```

If your layout differs, adjust paths in imports and `uvicorn` target.

---

## 16) License

Add your license of choice here.
