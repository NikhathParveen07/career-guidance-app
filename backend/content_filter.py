# ============================================
# backend/content_filter.py
# Stream boost and RIASEC boost logic
# Applied on top of raw semantic similarity scores
# ============================================


# ── Stream Boost ──────────────────────────────────────────────
STREAM_BOOST_VALUE = 1.2
# A 20% lift is applied to careers that match the student's stream.
# This reflects the structural reality of the Indian education system:
# stream-aligned careers are more immediately accessible.
# Tune this value (1.1–1.3) based on evaluation results.


def get_stream_boost(career_stream, student_stream):
    """
    Returns a multiplier based on whether the career's stream
    matches the student's declared Class 12 stream.

    Match   → 1.2x (20% boost)
    No match → 1.0x (no change)
    """
    if student_stream and career_stream == student_stream:
        return STREAM_BOOST_VALUE
    return 1.0


# ── RIASEC Boost ──────────────────────────────────────────────
# Boost values are calibrated to be meaningful but not overwhelming.
# Primary match (1.3x) is strong enough to elevate a relevant career
# but not so large that it overrides a clearly poor semantic match.
PRIMARY_RIASEC_BOOST   = 1.3
SECONDARY_RIASEC_BOOST = 1.15
NO_RIASEC_BOOST        = 1.0


def get_riasec_boost(career_row, riasec_top2):
    """
    Returns a multiplier based on how well the career's RIASEC
    profile aligns with the student's top-2 RIASEC types.

    Primary match   → 1.3x  (career RIASEC matches student's dominant type)
    Secondary match → 1.15x (career RIASEC matches student's secondary type)
    No match        → 1.0x  (no change — career still shown, just lower)

    Careers are never excluded — only re-ranked.
    This preserves the system's open career discovery philosophy.
    """
    primary   = career_row.get('primary_riasec',   '')
    secondary = career_row.get('secondary_riasec', '')

    student_primary   = riasec_top2[0]
    student_secondary = riasec_top2[1]

    # Check if career's RIASEC matches student's primary type
    if primary == student_primary or secondary == student_primary:
        return PRIMARY_RIASEC_BOOST

    # Check if career's RIASEC matches student's secondary type
    if primary == student_secondary or secondary == student_secondary:
        return SECONDARY_RIASEC_BOOST

    return NO_RIASEC_BOOST


# ── Cross-Stream Collaborative Penalty ───────────────────────
CROSS_STREAM_PENALTY = 0.5
# When SVD recommends careers from streams different from the
# student's stream, those collaborative scores are reduced by 50%.
# This prevents SVD from bleeding recommendations across stream
# boundaries while preserving its contribution within streams.


def apply_cross_stream_penalty(collab_score, career_stream, student_stream):
    """
    Reduce collaborative score for careers outside the student's stream.
    Applied before hybrid combination to prevent stream contamination.
    """
    if student_stream and career_stream != student_stream:
        return collab_score * CROSS_STREAM_PENALTY
    return collab_score
