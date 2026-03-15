# ============================================
# backend/riasec.py
# RIASEC quiz questions, scoring, and labels
# Based on Holland's Theory of Career Development
# ============================================


# ── Quiz Questions ────────────────────────────────────────────
# 12 questions — 2 per RIASEC type
# Each question is an activity-based self-assessment (rated 1–5)
# Pairs probe different facets of the same underlying trait
# to reduce response bias

RIASEC_QUESTIONS = [
    # Round 1 — direct activity preference
    {"q": "I enjoy building or fixing things with my hands",           "type": "R"},
    {"q": "I like researching facts and solving complex problems",      "type": "I"},
    {"q": "I enjoy drawing, writing, or creating original work",       "type": "A"},
    {"q": "I like teaching or helping others work through problems",   "type": "S"},
    {"q": "I enjoy leading teams or starting new initiatives",         "type": "E"},
    {"q": "I like organising data and following structured processes", "type": "C"},
    # Round 2 — orientation and preference framing
    {"q": "I prefer hands-on technical work over desk work",           "type": "R"},
    {"q": "I enjoy analysing patterns and thinking logically",         "type": "I"},
    {"q": "I express myself through music, art, or creative writing",  "type": "A"},
    {"q": "I find it rewarding to support and care for others",        "type": "S"},
    {"q": "I enjoy managing, negotiating, or competing",               "type": "E"},
    {"q": "I prefer clear rules, accuracy, and working with numbers",  "type": "C"},
]


# ── Labels and Descriptions ───────────────────────────────────
RIASEC_LABELS = {
    "R": "Realistic",
    "I": "Investigative",
    "A": "Artistic",
    "S": "Social",
    "E": "Enterprising",
    "C": "Conventional"
}

RIASEC_DESCRIPTIONS = {
    "R": "Hands-on, technical, physical work",
    "I": "Analytical, curious, research-driven",
    "A": "Creative, expressive, original",
    "S": "Caring, cooperative, people-focused",
    "E": "Ambitious, persuasive, leadership-oriented",
    "C": "Organised, detail-oriented, methodical"
}

RIASEC_FULL_DESCRIPTIONS = {
    "R": "You enjoy practical, hands-on activities. You prefer working with tools, machines, or physical materials over working with people or ideas.",
    "I": "You are naturally curious and analytical. You enjoy researching, experimenting, and solving complex intellectual problems.",
    "A": "You are creative and expressive. You thrive in environments that allow you to produce original work through art, writing, music, or design.",
    "S": "You are empathetic and cooperative. You enjoy helping, teaching, counselling, and working closely with other people.",
    "E": "You are ambitious and persuasive. You enjoy leading, managing, selling, and taking initiative in competitive environments.",
    "C": "You are organised and precise. You prefer structured tasks involving data, records, numbers, and well-defined procedures."
}


# ── Scoring ───────────────────────────────────────────────────
def compute_riasec_scores(answers):
    """
    Compute RIASEC scores from quiz answers.

    Args:
        answers: dict mapping question index (0–11) to rating (1–5)

    Returns:
        dict with keys:
            scores      — raw score per type (2–10 each)
            ranked      — list of (type, score) sorted descending
            top2        — list of top 2 type codes e.g. ['I', 'R']
            riasec_code — string e.g. 'IR'
    """
    scores = {"R": 0, "I": 0, "A": 0, "S": 0, "E": 0, "C": 0}

    for i, q in enumerate(RIASEC_QUESTIONS):
        scores[q['type']] += answers.get(i, 3)  # default 3 if missing

    ranked      = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top2        = [ranked[0][0], ranked[1][0]]
    riasec_code = "".join(top2)

    return {
        "scores":      scores,
        "ranked":      ranked,
        "top2":        top2,
        "riasec_code": riasec_code
    }
