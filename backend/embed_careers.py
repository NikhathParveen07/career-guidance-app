# ============================================
# backend/embed_careers.py
# Embed career list into Pinecone
# Called automatically after every O*NET refresh
# and after the India-specific CSV is merged in
# ============================================
import os
import time
from sentence_transformers import SentenceTransformer


def build_career_text(row):
    """
    Combine job fields into a single string for embedding.
    More context = better semantic search results.
    """
    parts = [
        row.get("job_title", ""),
        row.get("sector", ""),
        row.get("stream", row.get("12th_stream", "")),
        row.get("core_skills", ""),
    ]
    return " ".join(str(p) for p in parts if p)


def embed_and_upsert(df, pinecone_index, sentence_model, batch_size=100):
    """
    Embed all careers and upsert into Pinecone.

    Args:
        df              — merged career DataFrame (onet + india_specific)
        pinecone_index  — loaded Pinecone index object
        sentence_model  — loaded SentenceTransformer model
        batch_size      — number of vectors per upsert call (100 is safe)

    Each vector ID is "career_{row_index}" matching how hybrid_engine
    reads IDs: int(match['id'].split('_')[1])
    """
    print(f"Embedding {len(df)} careers into Pinecone...")

    vectors = []
    for i, row in df.iterrows():
        text      = build_career_text(row)
        embedding = sentence_model.encode(text).tolist()

        vectors.append({
            "id": f"career_{i}",
            "values": embedding,
            "metadata": {
                "job_title":        str(row.get("job_title", "")),
                "stream":           str(row.get("stream", row.get("12th_stream", ""))),
                "sector":           str(row.get("sector", "")),
                "primary_riasec":   str(row.get("primary_riasec", "")),
                "secondary_riasec": str(row.get("secondary_riasec", "")),
                "core_skills":      str(row.get("core_skills", "")),
            }
        })

    # Upsert in batches — Pinecone has a 2MB per request limit
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end   = start + batch_size
        batch = vectors[start:end]
        pinecone_index.upsert(vectors=batch)
        print(f"  Upserted batch {batch_num + 1}/{total_batches} ({len(batch)} vectors)")
        time.sleep(0.2)  # avoid rate limiting

    print(f"Pinecone index updated — {len(vectors)} vectors total")
