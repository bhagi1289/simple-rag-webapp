from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag import add_chunk, ask_llm, clear_chunks

app = FastAPI(title='RAG web App')

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5173"],
    # allow_credentails = True,
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
    clear_chunks()
    with open('../cat-facts.txt') as f:
        for line in f:
            add_chunk(line.strip())
    return {"status": "indexed"}

@app.post("/ask")
def ask(req: AskRequest):
    return ask_llm(req.question, top_k=req.top_k)