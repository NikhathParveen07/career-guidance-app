# ============================================
# backend/hybrid_engine.py
# Combines content-based and collaborative scores
# into a single ranked recommendation list.
#
# Adaptive weighting strategy (Do et al. [R8]):
#   Weights shift based on real interaction volume.
#   Cold start → pure content.
#   As real Keep/Drop data accumulates → collaborative
#   component earns progressively more influence.
# ============================================
import numpy as np
import requests
from backend.embeddings import search_careers
from backend.content_filter import (
    get_stream_boost,
    get_riasec_boost,
    apply_cross_stream_penalty
)


def get_adaptive_weights(n_real_interactions):
    """
    Compute content and collaborative weights adaptively
    based on how many real Keep/Drop interactions exist.

    Thresholds:
      0          → pure content (cold start, no real data)
      1–9        → mostly content, collab activates lightly
      10–29      → balanced hybrid
      30+        → collab earns equal footing with content

    Directly implements the adaptive combination principle
    from Do et al. [R8], grounding collaborative influence
    only when sufficient real peer behaviour is observed.
    """
    if n_real_interactions == 0:
        return 1.0, 0.0
    elif n_real_interactions < 10:
        return 0.85, 0.15
    elif n_real_interactions < 30:
        return 0.70, 0.30
    else:
        return 0.55, 0.45


def get_real_interaction_count(supabase):
    """
    Count total real Keep/Drop interactions in Supabase.
    Used to determine adaptive weights for this session.
    Returns 0 safely if Supabase is unavailable.
    """
    try:
        result = supabase.table("live_interactions").select("student_id", count="exact").execute()
        return result.count or 0
    except Exception:
        return 0


def expand_query(query, groq_key):
    """
    Expand vague student interest queries into richer semantic descriptions.
    Only runs if query is fewer than 15 words.
    Returns (expanded_query, was_expanded) tuple.
    """
    if len(query.split()) > 15:
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
            expanded = r.json()["choices"][0]["message"]["content"].strip()
            if expanded and expanded.lower() != query.lower():
                return expanded, True
    except Exception:
        pass

    return query, False


def get_collab_scores(user_id, n_careers, svd_model):
    """
    Compute and normalise collaborative scores for all careers.
    Normalisation maps raw SVD predictions to [0, 1] range.
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
        return {cid: 0.5 for cid in raw_scores}


def get_recommendations(user_id, query, student_stream, riasec_top2,
                        df, sentence_model, index, svd_model,
                        supabase, is_cold_start=True, top_k=10):
    """
    Full hybrid recommendation pipeline with adaptive weighting.
    Weights are determined by real Keep/Drop interaction volume.
    """
    n_real = get_real_interaction_count(supabase)
    content_weight, collab_weight = get_adaptive_weights(n_real)

    matches = search_careers(query, sentence_model, index, top_k=20)

    collab_scores = {}
    if collab_weight > 0:
        collab_scores = get_collab_scores(user_id, len(df), svd_model)

    results = []
    for match in matches:
        try:
            cid = int(match['id'].split('_')[1])
        except (ValueError, IndexError):
            continue

        if cid >= len(df):
            continue

        career_row    = df.iloc[cid]
        career_stream = match['metadata']['stream']
        content_score = match['score']

        collab_score = collab_scores.get(cid, 0.5) if collab_weight > 0 else 0.0
        collab_score = apply_cross_stream_penalty(
            collab_score, career_stream, student_stream
        )

        stream_boost = get_stream_boost(career_stream, student_stream)
        riasec_boost = get_riasec_boost(career_row.to_dict(), riasec_top2)

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
            'is_cold_start':    collab_weight == 0.0
        })

    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results[:top_k]
