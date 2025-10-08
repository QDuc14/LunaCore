from fastapi import FastAPI
from app.router import router

from app.rag.embedder import collection

app = FastAPI(title="Luna Core")
app.include_router(router)

# Sanity check: ensure the collection is created
print("Count:", collection.count())
got = collection.get(limit=3, include=["documents","metadatas"])
print("Keys:", got.keys())           # typically includes 'ids'
print("Sample IDs:", got.get("ids"))