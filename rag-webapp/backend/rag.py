import ollama
import math
from backend.db import get_conn

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

def embed_text(text: str):
    return ollama.embed(model=EMBEDDING_MODEL, input=text)["embeddings"][0]

def cosine_similarity(a,b):
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))

    return dot / (norm_a * norm_b)

def add_chunk(chunk: str):
    chunk = chunk.strip()
    if not chunk:
        return
    emb = embed_text(chunk)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rag_chunks (chunk, embedding) VALUES(%s, %s)",
                (chunk, emb)
            )

def clear_chunks():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE rag_chunks")

def retrieve(query: str, top_k: int=3):
    q_emb = embed_text(query)

    with get_conn() as conn:
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

            return cur.fetchall()

def ask_llm(question: str, top_k: int=3):
    retrieved = retrieve(question)

    context = "\n".join(
        [f" - {chunk}" for chunk, _ in retrieved]
    )

    system_prompt = f'''
                You are a helpful assistant.
                Use ONLY the following context to answer.
                If the answer is not in the context, say "I don't know." 

                Context: {context}  
    '''.strip()

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    return {
        "answer": stream["message"]["content"],
        "sources": [chunk for chunk, _ in retrieved]
    }