# ============================================
# backend/hybrid_engine.py
# Combines content-based and collaborative scores
# with adaptive cold-start weighting.
#
# Cold-start design:
#   - When a new student arrives with zero real interactions,
#     the system falls back to pure content-based retrieval.
#   - As real Keep/Drop signals accumulate in Supabase,
#     the collaborative weight grows across four stages.
#   - The is_cold_start flag is determined at runtime from
#     the actual interaction count, NOT hardcoded in app.py.
# ============================================
import numpy as np
import requests
from backend.embeddings import search_careers
from backend.content_filter import (
    get_stream_boost,
    get_riasec_boost,
    apply_cross_stream_penalty
)


# ── Adaptive weight schedule ──────────────────────────────────
# Four stages based on real interaction volume.
# Stage 0 (cold start): pure content retrieval — no collaborative signal.
# Stage 1–3: collaborative weight grows as real data accumulates.
# These thresholds are tuned to the expected interaction rate
# for a single Class 12 student session (typically 5–15 decisions).

WEIGHT_STAGES = [
    (1,  1.00, 0.00),   # Stage 0: cold start — content only (n = 0)
    (10, 0.85, 0.15),   # Stage 1: sparse
    (30, 0.70, 0.30),   # Stage 2: moderate
    (float('inf'), 0.55, 0.45),  # Stage 3: rich
]

def get_adaptive_weights(n_real_interactions):
    """
    Return (content_weight, collab_weight) based on how many real
    Keep/Drop interactions exist for this student.

    This implements the four-stage adaptive schedule described in the paper.
    The weights shift progressively so the collaborative component only
    contributes once it has enough signal to be reliable.

    Args:
        n_real_interactions: int — count of real interactions from Supabase

    Returns:
        (float, float) — content_weight, collab_weight that sum to 1.0
    """
    for threshold, cw, sw in WEIGHT_STAGES:
        if n_real_interactions < threshold:
            return cw, sw
    # Fallback — should never reach here given float('inf') sentinel
    return 0.55, 0.45


def get_real_interaction_count(supabase, student_id=None):
    """
    Count real Keep/Drop interactions in Supabase.

    If student_id is provided, count only that student's interactions
    (per-student cold start). Otherwise count globally.

    Returns 0 safely if Supabase is unavailable.
    """
    try:
        query = supabase.table("live_interactions").select(
            "student_id", count="exact"
        )
        if student_id:
            query = query.eq("student_id", student_id)
        result = query.execute()
        return result.count or 0
    except Exception:
        return 0


def expand_query(query, groq_key):
    """
    Expand a short or vague student interest query using Llama 3.3-70B
    hosted on Groq. Only triggered when the query is fewer than 15 words.

    The expansion rewrites the student's raw text into a richer semantic
    description that improves embedding similarity with career texts.

    Args:
        query: str — raw student interest text
        groq_key: str — Groq API key

    Returns:
        (expanded_query, was_expanded): tuple of (str, bool)
        Falls back to original query on any API failure.
    """
    if not query or not query.strip():
        return query, False

    # Only expand short queries — long ones already carry enough signal
    if len(query.split()) > 15:
        return query, False

    if not groq_key:
        return query, False

    prompt = (
        f'A Class 12 student wrote this about their interests:\n"{query}"\n\n'
        "Rewrite this in 2-3 simple sentences that sound like a student talking about "
        "what they enjoy doing in daily life. Use casual, friendly language. "
        "Talk about activities, hobbies, and things they like doing — not jobs or careers. "
        "Do NOT mention any job titles, professions, or career fields. "
        "Write as if the student is describing themselves to a friend.\n\n"
        "Return only the rewritten description. Nothing else."
    )

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            },
            json={
                "model":       "llama-3.3-70b-versatile",
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens":  150
            },
            timeout=10
        )
        if r.status_code == 200:
            data    = r.json()
            choices = data.get("choices", [])
            if choices:
                expanded = choices[0].get("message", {}).get("content", "").strip()
                if expanded and expanded.lower() != query.lower() and len(expanded) > 10:
                    return expanded, True
    except Exception as e:
        print(f"expand_query failed: {e}")

    return query, False


def get_collab_scores(user_id, n_careers, svd_model):
    """
    Compute normalised collaborative scores for all careers in [0, 1].

    Raw SVD predictions are in the range [1, 5]. We normalise them to
    [0, 1] so they are on the same scale as cosine similarity scores
    before the weighted combination.

    Args:
        user_id: str — student identifier
        n_careers: int — total number of careers in the index
        svd_model: LightSVD — trained model instance

    Returns:
        dict mapping career_id (int) -> normalised score (float)
    """
    raw_scores = {cid: svd_model.predict(user_id, cid) for cid in range(n_careers)}

    max_s = max(raw_scores.values())
    min_s = min(raw_scores.values())

    if max_s > min_s:
        return {
            cid: (score - min_s) / (max_s - min_s)
            for cid, score in raw_scores.items()
        }
    else:
        # All predictions identical — return neutral 0.5 for all
        return {cid: 0.5 for cid in raw_scores}


def get_recommendations(user_id, query, student_stream, riasec_top2,
                        df, sentence_model, index, svd_model,
                        supabase, top_k=10):
    """
    Full hybrid recommendation pipeline.

    Pipeline steps:
    1. Determine adaptive weights from real interaction count.
       If count == 0, operate in cold-start mode (content only).
    2. Retrieve top-20 candidates via semantic search (Pinecone).
    3. If not cold-start, compute normalised collaborative scores (LightSVD).
    4. Apply cross-stream penalty to collaborative scores for out-of-stream careers.
    5. Combine content and collaborative scores with adaptive weights.
    6. Apply stream boost (1.2×) and RIASEC boost (1.3×/1.15×/1.0×).
    7. Sort descending and return top_k results.

    The is_cold_start flag is determined here from actual interaction
    count — it is NOT passed in from app.py. This ensures the system
    automatically transitions out of cold-start mode as real data grows.

    Args:
        user_id: str — unique student session ID
        query: str — student interest text (possibly LLM-expanded)
        student_stream: str — Science / Commerce / Arts / Vocational
        riasec_top2: list — e.g. ['I', 'R']
        df: DataFrame — merged career dataset (O*NET + India-specific)
        sentence_model: SentenceTransformer
        index: Pinecone index
        svd_model: LightSVD
        supabase: Supabase client
        top_k: int — number of results to return

    Returns:
        list of recommendation dicts sorted by final_score descending
    """
    if not query or df.empty:
        return []

    # ── Step 1: Determine cold-start state from real data ─────
    # Count only this student's interactions for per-student cold start.
    # A new student always starts in cold-start mode regardless of how
    # many other students have interacted with the system.
    n_real       = get_real_interaction_count(supabase, student_id=user_id)
    is_cold_start = (n_real == 0)
    content_weight, collab_weight = get_adaptive_weights(n_real)

    # ── Step 2: Semantic retrieval (always runs) ──────────────
    try:
        matches = search_careers(query, sentence_model, index, top_k=20)
    except Exception as e:
        print(f"Pinecone search failed: {e}")
        return []

    if not matches:
        return []

    # ── Step 3: Collaborative scores (skipped in cold start) ──
    collab_scores = {}
    if not is_cold_start and collab_weight > 0:
        try:
            collab_scores = get_collab_scores(user_id, len(df), svd_model)
        except Exception as e:
            print(f"Collab scoring failed, falling back to content only: {e}")
            content_weight = 1.0
            collab_weight  = 0.0

    # ── Steps 4–7: Score combination and re-ranking ───────────
    results = []
    for match in matches:
        try:
            cid = int(match['id'].split('_')[1])
        except (ValueError, IndexError):
            continue

        if cid >= len(df):
            continue

        career_row    = df.iloc[cid]
        career_stream = match['metadata'].get('stream', '')
        content_score = match['score']

        # Apply cross-stream penalty to collaborative score before combining
        collab_score = collab_scores.get(cid, 0.5) if (not is_cold_start and collab_weight > 0) else 0.0
        collab_score = apply_cross_stream_penalty(
            collab_score, career_stream, student_stream
        )

        # stream_boost = get_stream_boost(career_stream, student_stream)
        # riasec_boost = get_riasec_boost(career_row.to_dict(), riasec_top2)
        stream_boost = 1.0  # BASELINE A: no boost for test chnage later
        riasec_boost = 1.0  # BASELINE A: no boost

        # Weighted combination then post-score multipliers
        base_score  = content_weight * content_score + collab_weight * collab_score
        final_score = base_score * stream_boost * riasec_boost

        results.append({
            'career_id':        cid,
            'career':           career_row['job_title'],
            'stream':           career_stream,
            'sector':           career_row.get('sector', ''),
            'primary_riasec':   career_row.get('primary_riasec', ''),
            'secondary_riasec': career_row.get('secondary_riasec', ''),
            'core_skills':      career_row.get('core_skills', ''),
            'final_score':      round(final_score,   4),
            'content_score':    round(content_score, 4),
            'collab_score':     round(collab_score,  4),
            'stream_boost':     stream_boost,
            'riasec_boost':     riasec_boost,
            'is_cold_start':    is_cold_start,
            'n_real_interactions': n_real,
            'content_weight':   content_weight,
            'collab_weight':    collab_weight,
        })

    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results[:top_k]
