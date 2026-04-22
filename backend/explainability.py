# ============================================
# backend/explainability.py
# Generate plain-English explanations for each recommendation
# Three components: interest match, personality fit, stream fit
# ============================================
from backend.riasec import RIASEC_LABELS


# ── Stop words to ignore in keyword matching ──────────────────
STOP_WORDS = {
    'and', 'or', 'the', 'a', 'an', 'in', 'of', 'to',
    'with', 'for', 'is', 'i', 'like', 'enjoy', 'love',
    'want', 'am', 'my', 'me', 'do', 'at', 'by', 'on'
}


def explain_interest_match(query, core_skills, content_score):
    """
    Component 1: Why this career matches the student's interest text.

    Method:
    - Tokenise the student's query and the career's core skills
    - Find overlapping keywords (ignoring stop words)
    - If overlap found: name the matching concepts explicitly
    - If no overlap: report semantic similarity percentage
      (cosine similarity from Pinecone, converted to %)

    This component grounds the explanation in the student's
    own words wherever possible.
    """
    
    query_words = set(query.lower().split()) - STOP_WORDS
    skill_words = set(str(core_skills or '').lower().replace(',', '').split()) - STOP_WORDS
    overlap     = query_words & skill_words

    if overlap:
        keywords = ', '.join(sorted(overlap))
        return f"Your interest in '{keywords}' directly matches this career's core skills."
    elif content_score >= 0.5:
        return f"Strong semantic alignment ({content_score:.0%}) between your interests and this career's skill profile."
    elif content_score >= 0.35:
        return f"Moderate alignment ({content_score:.0%}) between your interests and this career."
    else:
        return f"Broad alignment ({content_score:.0%}) — this career expands your interest area."


def explain_riasec_fit(riasec_top2, primary_riasec, secondary_riasec, riasec_boost):
    """
    Component 2: How well the career's RIASEC codes match the student's profile.

    Three explanation levels:
    - Strong fit  (1.3x boost): career matches student's dominant type
    - Moderate fit (1.15x boost): career matches student's secondary type
    - Growth opportunity (1.0x): no RIASEC overlap — framed constructively

    Careers with no RIASEC match are never described negatively.
    The 'growth opportunity' framing preserves open career exploration.
    """
    student_primary   = riasec_top2[0]
    student_secondary = riasec_top2[1]

    primary_label   = RIASEC_LABELS.get(student_primary,   student_primary)
    secondary_label = RIASEC_LABELS.get(student_secondary, student_secondary)
    career_label    = RIASEC_LABELS.get(primary_riasec,    primary_riasec)

    if riasec_boost == 1.3:
        return (f"Strong personality fit — your dominant {primary_label} "
                f"trait aligns with this career's core profile.")
    elif riasec_boost == 1.15:
        return (f"Moderate personality fit — your {secondary_label} "
                f"trait fits this career's work style.")
    else:
        return (f"Growth opportunity — this career suits {career_label} "
                f"profiles and may broaden your professional identity.")


def explain_stream_fit(career_stream, student_stream, stream_boost):
    """
    Component 3: Stream alignment between the student and the career.

    For stream-aligned careers: confirms direct accessibility.
    For other-stream careers: acknowledges the gap constructively
    and notes that bridging is possible — not a barrier message.
    """
    if stream_boost > 1.0:
        return (f"Directly aligned with your {student_stream} stream — "
                f"standard Class 12 pathway applies.")
    else:
        return (f"This career is from the {career_stream} stream — "
                f"accessible through bridging courses or entrance exams.")


def generate_explanation(rec, query, riasec_top2, student_stream):
    """
    Master function — combines all 3 explanation components
    for a single recommendation.

    Args:
        rec           — recommendation dict from hybrid_engine
        query         — student's free-text interest query
        riasec_top2   — list of student's top 2 RIASEC type codes
        student_stream — student's Class 12 stream

    Returns:
        tuple of (interest_match, riasec_fit, stream_fit) strings
    """
    interest = explain_interest_match(
        query,
        rec['core_skills'],
        rec['content_score']
    )

    personality = explain_riasec_fit(
        riasec_top2,
        rec['primary_riasec'],
        rec['secondary_riasec'],
        rec['riasec_boost']
    )

    stream = explain_stream_fit(
        rec['stream'],
        student_stream,
        rec['stream_boost']
    )

    return interest, personality, stream
