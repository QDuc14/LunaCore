# app/services/llm_provider.py
from __future__ import annotations
import os, json, asyncio
from typing import Any, Dict, AsyncIterator, List
from dotenv import load_dotenv
load_dotenv(override=False) 

# Env switches
PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").lower()  # "openai_compatible" or "ollama"

# OpenAI-compatible (managed)
from openai import AsyncOpenAI
_OAI_BASE = os.getenv("LLM_API_BASE", "").strip()  # e.g. https://api.groq.com/openai/v1
_OAI_KEY  = os.getenv("LLM_API_KEY", "").strip()
_EXTRA_HEADERS = json.loads(os.getenv("LLM_EXTRA_HEADERS", "{}"))
_OAI_CLIENT = AsyncOpenAI(base_url=_OAI_BASE or None, api_key=_OAI_KEY or None, default_headers=_EXTRA_HEADERS or None)

EMBED_API_BASE = os.getenv("EMBED_API_BASE", os.getenv("LLM_API_BASE", "")).strip()
EMBED_API_KEY  = os.getenv("EMBED_API_KEY",  os.getenv("LLM_API_KEY", "")).strip()
_EMBED_CLIENT = AsyncOpenAI(base_url=EMBED_API_BASE or None,
                            api_key=EMBED_API_KEY or None)

# Models (override via env)
CHAT_MODEL  = os.getenv("LLM_CHAT_MODEL", "llama-3.1-8b-instruct")
EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "text-embedding-3-small")  # or "nomic-embed-text" if your provider supports it

# Ollama fallback (local or remote)
import ollama
_OLLAMA_DEFAULTS = {
    "keep_alive": os.getenv("LUNA_KEEP_ALIVE", "30m"),
    "num_thread": os.cpu_count() or 4,
    "num_ctx": int(os.getenv("LUNA_NUM_CTX", "4096")),
    "num_predict": int(os.getenv("LUNA_NUM_PREDICT", "256")),
    "temperature": float(os.getenv("LUNA_TEMPERATURE", "0.2")),
    "top_k": int(os.getenv("LUNA_TOP_K", "30")),
    "top_p": float(os.getenv("LUNA_TOP_P", "0.9")),
}

def _ollama_opts(user: Dict[str, Any] | None) -> Dict[str, Any]:
    out = dict(_OLLAMA_DEFAULTS)
    if user:
        out.update({k: v for k, v in user.items() if v is not None})
    return out

# --------- Unified API ---------
async def embed_texts(texts: List[str]) -> List[List[float]]:
    if PROVIDER == "openai_compatible":
        resp = await _EMBED_CLIENT.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    # ollama
    def _call():
        return [ollama.embeddings(model=os.getenv("EMBED_MODEL","nomic-embed-text"), prompt=t)["embedding"] for t in texts]
    return await asyncio.to_thread(_call)

async def chat_once(messages: list[dict], *, options: Dict[str, Any] | None = None, model: str | None = None) -> str:
    model = model or CHAT_MODEL
    if PROVIDER == "openai_compatible":
        rsp = await _OAI_CLIENT.chat.completions.create(model=model, messages=messages, temperature=0.7)
        return (rsp.choices[0].message.content or "")
    # ollama
    def _call():
        res = ollama.chat(model=model, messages=messages, options=_ollama_opts(options))
        return res.get("message", {}).get("content", "")
    return await asyncio.to_thread(_call)

async def chat_stream(messages: list[dict], *, options: Dict[str, Any] | None = None, model: str | None = None) -> AsyncIterator[str]:
    model = model or CHAT_MODEL
    if PROVIDER == "openai_compatible":
        stream = await _OAI_CLIENT.chat.completions.create(model=model, messages=messages, stream=True, temperature=0.7)
        async for ev in stream:
            d = ev.choices[0].delta
            if d and d.content:
                yield d.content
        return
    # ollama streaming: wrap thread
    import threading, queue
    q: "queue.Queue[tuple[str,str] | object]" = queue.Queue()
    SENTINEL = object()
    def worker():
        try:
            parts = []
            for chunk in ollama.chat(model=model, messages=messages, options=_ollama_opts(options), stream=True):
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    parts.append(delta)
                    q.put(("delta", delta))
            q.put(("done", "".join(parts)))
        except Exception as e:
            q.put(("error", str(e)))
        finally:
            q.put(SENTINEL)
    threading.Thread(target=worker, daemon=True).start()
    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is SENTINEL:
            break
        kind, payload = item  # type: ignore
        if kind == "delta":
            yield payload
        elif kind == "error":
            raise RuntimeError(payload)
        elif kind == "done":
            return
