
import faiss
import numpy as np
from embedder.chunk_and_embed import embed_question


def create_faiss_index(embeddings):

    vectors = np.asarray(embeddings, dtype="float32")

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Embeddings are empty or invalid.")

    # Embeddings are already normalized
    index = faiss.IndexFlatIP(vectors.shape[1])

    index.add(vectors)

    return index


def search_faiss_index(index, question, chunks, top_k=4):

    if not chunks:
        return []

    query_vector = embed_question(question)

    k = min(top_k, len(chunks))

    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if 0 <= idx < len(chunks):
            results.append({
                "text": chunks[int(idx)],
                "score": float(score)
            })

    return results
