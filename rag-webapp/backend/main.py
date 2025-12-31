from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag import add_chunk, ask_llm, clear_chunks
from backend.logger import logger
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "cat-facts.txt"   # adjust if needed

print(DATA_FILE)
app = FastAPI(title='RAG web App')

origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

class AskRequest(BaseModel):
    question: str
    top_k: int = 3

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/index")
def index_data():
    logger.info("Indexing started")
    clear_chunks()

    count = 0
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            parts = text.split(". ")
            for p in parts:
                p = p.strip()
                if p:
                    add_chunk(p)
                    count += 1

    logger.info(f"Indexing completed. Total chunks indexed: {count}")
    return {"status": "indexed"}

@app.post("/ask")
def ask(req: AskRequest):
    logger.info(f"/ask called | question='{req.question}' | top_k={req.top_k}")
    return ask_llm(req.question, top_k=req.top_k)