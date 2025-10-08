from __future__ import annotations

import os
import asyncio
from threading import Thread
from queue import Queue, Empty
from typing import Any, Dict, Iterable

import ollama

# Add these near the top:
_DEFAULT_OPTS = {
    "keep_alive": os.getenv("LUNA_KEEP_ALIVE", "30m"),
    "num_thread": os.cpu_count() or 4,
    "num_ctx": int(os.getenv("LUNA_NUM_CTX", "4096")),
    "num_predict": int(os.getenv("LUNA_NUM_PREDICT", "256")),
    "temperature": float(os.getenv("LUNA_TEMPERATURE", "0.2")),
    "top_k": int(os.getenv("LUNA_TOP_K", "30")),
    "top_p": float(os.getenv("LUNA_TOP_P", "0.9")),
    # ===================================================================
    # "num_ctx": int(os.getenv("LUNA_NUM_CTX", "2048")),
    # "num_thread": max(2, (os.cpu_count() or 4) - 1), 
    # "num_batch": int(os.getenv("LUNA_NUM_BATCH", "128")),
}

_LIGHTWEIGHT_OPTS = {
    "keep_alive": os.getenv("LUNA_KEEP_ALIVE", "15m"),   # warm model to avoid cold loads
    "num_thread": max(2, (os.cpu_count() or 4) - 1),     # use CPU well without contention
    "num_ctx": int(os.getenv("LUNA_NUM_CTX", "1024")),   # <<< big win: shrink KV cache
    "num_predict": int(os.getenv("LUNA_NUM_PREDICT", "220")),
    "temperature": float(os.getenv("LUNA_TEMPERATURE", "0.2")),
    "top_k": int(os.getenv("LUNA_TOP_K", "40")),         # slight bump is fine
    "top_p": float(os.getenv("LUNA_TOP_P", "0.9")),
    "num_batch": int(os.getenv("LUNA_NUM_BATCH", "128")),  # lower = safer on low VRAM
}

def opts_for_prompt(prompt_token_estimate: int) -> dict:
    # start with your defaults
    opts = dict(_LIGHTWEIGHT_OPTS)
    if prompt_token_estimate < 600:
        opts["num_ctx"] = 768
        opts["num_predict"] = min(_LIGHTWEIGHT_OPTS.get("num_predict", 220), 200)
    elif prompt_token_estimate < 1200:
        opts["num_ctx"] = 1024
    else:
        opts["num_ctx"] = 1536  # only when you truly need it
    return opts


def _merge_opts(user: Dict[str, Any] | None) -> Dict[str, Any]:
    out = dict(_DEFAULT_OPTS)
    if user:
        out.update({k: v for k, v in user.items() if v is not None})
    return out


async def chat_with_ollama(model: str, messages: list[dict], options: Dict[str, Any] | None = None) -> str:
    opts = _merge_opts(options)
    def _call() -> str:
        res = ollama.chat(model=model, messages=messages, options=opts)
        # shape: {"message": {"content": "..."}}
        return res.get("message", {}).get("content", "")
    return await asyncio.to_thread(_call)


async def stream_chat_with_ollama(model: str, messages: list[dict], options: Dict[str, Any] | None = None):
    """
    Async generator yielding {"delta": "..."} and finally {"done": True, "text": "..."}.
    Runs the blocking ollama.chat(..., stream=True) in a background thread.
    """
    opts = _merge_opts(options)
    q: Queue = Queue()
    SENTINEL = object()

    def worker():
        try:
            acc_parts = []
            for chunk in ollama.chat(model=model, messages=messages, options=opts, stream=True):
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    acc_parts.append(delta)
                    q.put(("delta", delta))
            q.put(("done", "".join(acc_parts)))
        except Exception as e:
            q.put(("error", str(e)))
        finally:
            q.put(SENTINEL)

    Thread(target=worker, daemon=True).start()
    loop = asyncio.get_running_loop()

    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is SENTINEL:
            break
        kind, payload = item
        if kind == "delta":
            yield {"delta": payload}
        elif kind == "done":
            yield {"done": True, "text": payload}
        elif kind == "error":
            # Let your router catch and convert to SSE error
            raise RuntimeError(payload)