from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_texts(pages, chunk_size=180, overlap=30):
    """Split Wikipedia page text into overlapping word chunks."""
    chunks = []
    step = max(1, chunk_size - overlap)
    for page in pages:
        words = page["text"].split()
        for start in range(0, len(words), step):
            part = " ".join(words[start:start + chunk_size]).strip()
            if part:
                chunks.append(part)
            if start + chunk_size >= len(words):
                break
    return chunks

def embed_chunks(chunks):
    vectors = get_embedding_model().encode(
        chunks, convert_to_numpy=True, normalize_embeddings=True
    )
    return np.asarray(vectors, dtype="float32")

def embed_question(question):
    vector = get_embedding_model().encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    )
    return np.asarray(vector, dtype="float32")
