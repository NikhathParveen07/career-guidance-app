# ============================================
# backend/hybrid_engine.py
# Combines content-based and collaborative scores
# into a single ranked recommendation list
# ============================================
import numpy as np
from backend.embeddings import search_careers
from backend.content_filter import (
    get_stream_boost,
    get_riasec_boost,
    apply_cross_stream_penalty
)


# ── Hybrid Weights ────────────────────────────────────────────
CONTENT_WEIGHT = 0.70
COLLAB_WEIGHT  = 0.30
# 70% content-based (semantic similarity) + 30% collaborative (SVD)
# Content weight is higher because semantic signals are more reliable
# at the current dataset scale. As real interaction data accumulates,
# increasing the collaborative weight is recommended.


def get_collab_scores(user_id, n_careers, svd_model):
    """
    Compute and normalise collaborative scores for all careers.
    Normalisation maps raw SVD predictions to [0, 1] range
    so they are comparable with cosine similarity scores.
    """
    raw_scores = {}
    for cid in range(n_careers):
        raw_scores[cid] = svd_model.predict(user_id, cid)

    max_s = max(raw_scores.values())
    min_s = min(raw_scores.values())

    if max_s > min_s:
        return {
            cid: (score - min_s) / (max_s - min_s)
            for cid, score in raw_scores.items()
        }
    else:
        # All scores are identical — return neutral 0.5
        return {cid: 0.5 for cid in raw_scores}


def get_recommendations(user_id, query, student_stream, riasec_top2,
                        df, sentence_model, index, svd_model,
                        is_cold_start=True, top_k=10):
    """
    Full hybrid recommendation pipeline.

    Steps:
    1. Semantic search — retrieve top 20 careers by cosine similarity
    2. Collaborative scores — SVD predictions for all careers
    3. Cross-stream penalty — reduce collab score for wrong-stream careers
    4. Hybrid combination — weighted sum of content + collaborative
    5. Stream boost — 1.2x lift for stream-aligned careers
    6. RIASEC boost — 1.3x or 1.15x lift for personality-matched careers
    7. Sort and return top_k results

    Cold-start fallback:
    If the student has no interaction history, collab_weight is set to 0
    and the system falls back to pure content + psychometric scoring.
    """

    # Adjust weights for cold-start students
    content_weight = 1.0 if is_cold_start else CONTENT_WEIGHT
    collab_weight  = 0.0 if is_cold_start else COLLAB_WEIGHT

    # Step 1 — Semantic retrieval from Pinecone
    matches = search_careers(query, sentence_model, index, top_k=20)

    # Step 2 — Collaborative scores (skip if cold-start)
    collab_scores = {}
    if collab_weight > 0:
        collab_scores = get_collab_scores(user_id, len(df), svd_model)

    # Step 3–7 — Score each retrieved career
    results = []
    for match in matches:
        # Parse career index from Pinecone vector ID (format: "career_N")
        try:
            cid = int(match['id'].split('_')[1])
        except (ValueError, IndexError):
            continue

        if cid >= len(df):
            continue

        career_row    = df.iloc[cid]
        career_stream = match['metadata']['stream']
        content_score = match['score']

        # Get collaborative score with cross-stream penalty applied
        collab_score = collab_scores.get(cid, 0.5) if collab_weight > 0 else 0.0
        collab_score = apply_cross_stream_penalty(
            collab_score, career_stream, student_stream
        )

        # Compute boost factors
        stream_boost = get_stream_boost(career_stream, student_stream)
        riasec_boost = get_riasec_boost(career_row.to_dict(), riasec_top2)

        # Final hybrid score
        base_score  = content_weight * content_score + collab_weight * collab_score
        final_score = base_score * stream_boost * riasec_boost

        results.append({
            'career_id':        cid,
            'career':           career_row['job_title'],
            'stream':           career_stream,
            'sector':           career_row['sector'],
            'primary_riasec':   career_row['primary_riasec'],
            'secondary_riasec': career_row['secondary_riasec'],
            'core_skills':      career_row['core_skills'],
            'final_score':      round(final_score,   4),
            'content_score':    round(content_score, 4),
            'collab_score':     round(collab_score,  4),
            'stream_boost':     stream_boost,
            'riasec_boost':     riasec_boost,
            'is_cold_start':    is_cold_start
        })

    # Sort by final score descending and return top_k
    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results[:top_k]
