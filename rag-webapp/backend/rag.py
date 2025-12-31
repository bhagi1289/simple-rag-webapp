import ollama
import math
import os
from openai import OpenAI
from backend.db import get_conn
from backend.logger import logger
import numpy as np
import requests
from huggingface_hub import InferenceClient
from pgvector.psycopg import register_vector


EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

client = InferenceClient(provider="hf-inference", api_key=os.environ["HF_TOKEN"])

def embed_text(text: str):
    # HF feature-extraction usually returns token embeddings (2D). We mean-pool to get sentence embedding (1D).
    out = client.feature_extraction(text, model=HF_EMBED_MODEL)

    arr = np.asarray(out, dtype=np.float32)

    # If we got token embeddings: (seq_len, dim) -> mean pool -> (dim,)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)

    # Optional but recommended for cosine similarity search (normalize)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm

    return arr.tolist() 


def cosine_similarity(a,b):
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))

    return dot / (norm_a * norm_b)

def add_chunk(chunk: str):
    chunk = chunk.strip()
    if not chunk:
        return
    
    logger.info(f"Indexing chunk: {chunk[:80]}")

    emb = embed_text(chunk)
    with get_conn() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rag_chunks (chunk, embedding) VALUES(%s, %s)",
                (chunk, emb)
            )
        conn.commit()

def clear_chunks():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE rag_chunks")
        conn.commit()

def retrieve(query: str, top_k: int=3):
    q_emb = embed_text(query)

    with get_conn() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                '''
                    SELECT chunk, (1- (embedding <=> %s::vector)) AS similarity
                    FROM rag_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                ''',
                (q_emb, q_emb, top_k)
            )

            rows = cur.fetchall()
            logger.info(f"Retrieved {len(rows)} chunks for query: '{query}'")
            return rows

def ask_llm(question: str, top_k: int=3):
    retrieved = retrieve(question, top_k=max(top_k, 20))
    logger.info(f"User question: {question}")
    logger.info(f"Retrieved {len(retrieved)} raw chunks")

    for c, s in retrieved[:5]:
        logger.info(f"SIM={s:.3f} | {c[:80]}")

    MIN_SIM = 0.68
    filtered = [(c, s) for c, s in retrieved if s >= MIN_SIM]

    logger.info(f"{len(filtered)} chunks passed similarity threshold")

    # if nothing relevant, return "I don't know"
    if not filtered:
        return {
            "answer": "I don't know.",
            "sources": [{"chunk": c, "similarity": float(s)} for c, s in retrieved[:top_k]],
        }

    # use the filtered ones as context
    retrieved = filtered[:top_k]

    context = "\n".join(
        [f" - {chunk}" for chunk, _ in retrieved]
    )

    system_prompt = f'''
                You are a helpful assistant.
                Use ONLY the following context to answer.
                If the answer is not in the context, say "I don't know." 

                Context: 
                {context}  
            '''.strip()

    if not GROQ_API_KEY:
        return {
            "answer": "GROQ_API_KEY not configured on server.",
            "sources": [{"chunk": c, "similarity": float(s)} for c, s in retrieved],
        }
    
    logger.info("Sending request to Groq LLM")

    completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        )
    answer_text = completion.choices[0].message.content
    
    logger.info("Groq response received successfully")

    return {
        "answer": answer_text,
        "sources": [{"chunk": c, "similarity": float(s)} for c, s in retrieved],
    }