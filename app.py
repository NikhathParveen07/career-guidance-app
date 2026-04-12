# ============================================
# app.py — UI only, all logic lives in backend/
#
# Cold-start note: get_recommendations() now determines
# cold-start state internally from the real interaction
# count in Supabase. Do NOT pass is_cold_start here.
# ============================================
import streamlit as st
import uuid

from backend.data_loader    import load_careers, load_sentence_model, load_pinecone_index, load_supabase
from backend.collaborative  import load_svd_model, save_interaction
from backend.hybrid_engine  import get_recommendations, expand_query
from backend.riasec         import RIASEC_QUESTIONS, RIASEC_LABELS, RIASEC_DESCRIPTIONS, compute_riasec_scores
from backend.explainability import generate_explanation
from backend.job_market     import fetch_full_market_data
from backend.pathway        import fetch_career_pathway, fetch_local_recommendations


st.set_page_config(
    page_title="CareerCompass India", page_icon="🧭",
    layout="wide", initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 3rem 2rem; border-radius: 16px;
        text-align: center; color: white; margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.5rem; font-weight: 700; margin: 0; color: white; }
    .hero p  { font-size: 1.1rem; opacity: 0.85; margin-top: 0.5rem; color: white; }

    .career-card {
        background: #1e2530; border-radius: 12px; padding: 1.2rem 1.5rem;
        margin-bottom: 0.4rem; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border-left: 4px solid #e94560; color: #f0f0f0;
    }
    .career-card-kept {
        background: #1a2e1e; border-radius: 12px; padding: 1.2rem 1.5rem;
        margin-bottom: 0.4rem; box-shadow: 0 2px 12px rgba(34,197,94,0.15);
        border-left: 4px solid #22c55e; color: #f0f0f0;
    }
    .career-card-dropped {
        background: #181e28; border-radius: 12px; padding: 1.2rem 1.5rem;
        margin-bottom: 0.4rem; box-shadow: none;
        border-left: 4px solid #374151; color: #9ca3af;
        opacity: 0.55;
    }

    .keep-drop-divider {
        border: none; border-top: 1px solid #2e3a4e; margin: 0.75rem 0 0.5rem;
    }

    .status-pill-kept {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        background: #14532d44; color: #22c55e;
        font-size: 11px; font-weight: 700; letter-spacing: 0.05em;
        border: 1px solid #22c55e44;
    }
    .status-pill-dropped {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        background: #1f2937; color: #6b7280;
        font-size: 11px; font-weight: 600; letter-spacing: 0.05em;
    }
    .status-pill-none {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        background: transparent; color: #4b5563;
        font-size: 11px; letter-spacing: 0.05em;
    }

    .section-header-kept {
        font-size: 11px; color: #22c55e; text-transform: uppercase;
        letter-spacing: 0.12em; font-weight: 700;
        margin: 1.4rem 0 0.5rem; padding: 4px 14px;
        background: #14532d22; border-radius: 20px;
        display: inline-block; border: 1px solid #22c55e33;
    }
    .section-header-neutral {
        font-size: 11px; color: #60a5fa; text-transform: uppercase;
        letter-spacing: 0.12em; font-weight: 700;
        margin: 1.4rem 0 0.5rem; padding: 4px 14px;
        background: #1e3a5f22; border-radius: 20px;
        display: inline-block; border: 1px solid #60a5fa33;
    }
    .section-header-dropped {
        font-size: 11px; color: #4b5563; text-transform: uppercase;
        letter-spacing: 0.12em; font-weight: 700;
        margin: 1.4rem 0 0.5rem; padding: 4px 14px;
        background: #1f293722; border-radius: 20px;
        display: inline-block; border: 1px solid #37415133;
    }

    .expand-box {
        background: #1a2a3a; border-radius: 10px; padding: 1rem 1.2rem;
        border: 1px solid #2a4a6a; margin: 0.8rem 0;
        font-size: 0.95rem; color: #d0e8f8; line-height: 1.7;
    }
    .expand-label {
        font-size: 11px; color: #60a5fa; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 6px; font-weight: 600;
    }
    .expand-original { color: #8899aa; font-style: italic; margin-bottom: 10px; }
    .expand-result   { color: #d0e8f8; font-size: 0.95rem; line-height: 1.7; }

    .mkt-card {
        background: #1e2530; border-radius: 12px; padding: 1.2rem 1.5rem;
        border: 1px solid #2e3a4e; margin-bottom: 1rem;
    }
    .mkt-card-label {
        font-size: 11px; color: #8899aa; text-transform: uppercase;
        letter-spacing: 0.05em; margin: 0 0 10px;
    }
    .sal-card {
        background: #151e2a; border-radius: 10px; padding: 1rem;
        border: 1px solid #2e3a4e; height: 100%;
    }
    .sal-card.highlight { border-color: #3b82f6; }
    .sal-card p  { margin: 0; }
    .sal-label   { font-size: 11px; color: #8899aa; margin-bottom: 4px !important; }
    .sal-value   { font-size: 20px; font-weight: 600; color: #f0f0f0; margin-bottom: 4px !important; }
    .sal-value.blue { color: #60a5fa; }
    .sal-explain { font-size: 12px; color: #8899aa; line-height: 1.5; margin-bottom: 4px !important; }
    .sal-source  { font-size: 11px; color: #5a6a7a; margin-top: 4px !important; }

    .list-item {
        display: flex; align-items: center; gap: 10px;
        padding: 6px 0; border-bottom: 1px solid #1e2a38;
    }
    .list-item:last-child { border-bottom: none; }
    .list-num  { font-size: 12px; color: #5a6a7a; min-width: 18px; }
    .list-name { font-size: 14px; color: #d0d8e8; }

    .badge-pill {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 500; margin-right: 6px;
    }
    .badge-growing    { background: #1a3a2a; color: #86efac; }
    .badge-declining  { background: #3a1a1a; color: #fca5a5; }
    .badge-stable     { background: #1e2a3a; color: #93c5fd; }
    .badge-govbacked  { background: #2a2a1a; color: #fcd34d; }
    .badge-moderate   { background: #2a2010; color: #fbbf24; }
    .badge-competitive{ background: #3a1a1a; color: #fca5a5; }
    .badge-low        { background: #1a3a2a; color: #86efac; }

    .explain-text {
        font-size: 14px; color: #c0c8d8; line-height: 1.7;
        font-style: italic; margin: 8px 0 6px;
        padding-left: 12px; border-left: 2px solid #2e3a4e;
    }
    .source-text { font-size: 11px; color: #5a6a7a; }

    .metric-box {
        background: #1e2530; border-radius: 10px; padding: 1rem;
        text-align: center; border: 1px solid #2e3a4e;
    }
    .metric-box h3 { font-size: 1.6rem; color: #60a5fa; margin: 0; }
    .metric-box p  { font-size: 0.8rem; color: #aaaaaa; margin: 0; }

    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin-right: 4px;
    }
    .badge-science    { background: #1e3a5f; color: #93c5fd; }
    .badge-commerce   { background: #1a3a2a; color: #86efac; }
    .badge-arts       { background: #3a1a2e; color: #f9a8d4; }
    .badge-vocational { background: #3a2e1a; color: #fcd34d; }

    .why-box {
        background: #2a2510; border-radius: 8px; padding: 0.8rem 1rem;
        margin-top: 0.5rem; border: 1px solid #5a4a20;
        font-size: 0.85rem; color: #e5d5a0; line-height: 1.6;
    }
    .why-box b { color: #fcd34d; }

    .coldstart-banner {
        background: #1a2a3a; border: 1px solid #2a4a6a;
        border-radius: 8px; padding: 0.6rem 1rem;
        font-size: 12px; color: #93c5fd; margin-bottom: 0.8rem;
    }

    .step-box {
        background: #1e2530; border-radius: 8px; padding: 0.8rem 1rem;
        margin: 0.4rem 0; border-left: 3px solid #60a5fa;
        color: #d0d8e8; line-height: 1.5;
    }
    .exam-box {
        background: #1e2530; border-radius: 8px; padding: 0.8rem 1rem;
        margin: 0.4rem 0; border-left: 3px solid #e94560;
        color: #d0d8e8; line-height: 1.5;
    }
    .course-box {
        background: #1e2530; border-radius: 8px; padding: 0.8rem 1rem;
        margin: 0.4rem 0; border-left: 3px solid #fcd34d;
        color: #d0d8e8; line-height: 1.5;
    }
    .skill-box {
        background: #1e2530; border-radius: 8px; padding: 0.8rem 1rem;
        margin: 0.4rem 0; border-left: 3px solid #86efac;
        color: #d0d8e8; line-height: 1.5;
    }
    .scholarship-box {
        background: #1e2530; border-radius: 8px; padding: 0.8rem 1rem;
        margin: 0.4rem 0; border-left: 3px solid #a78bfa;
        color: #d0d8e8; line-height: 1.5;
    }
    .section-divider {
        border: none; border-top: 1px solid #2e3a4e; margin: 1.5rem 0;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #0f3460, #e94560);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 2rem; font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

STREAM_BADGE = {
    "Science": "badge-science", "Commerce": "badge-commerce",
    "Arts": "badge-arts",       "Vocational": "badge-vocational"
}
TREND_BADGE = {
    "Rapidly Growing": "badge-growing", "Growing": "badge-growing",
    "Stable": "badge-stable",           "Declining": "badge-declining",
    "Uncertain": "badge-stable",        "Insufficient Data": "badge-stable"
}
COMPETITION_BADGE = {
    "Low": "badge-low", "Moderate": "badge-moderate",
    "Competitive": "badge-competitive", "Very Competitive": "badge-competitive"
}
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi",
    "Jammu & Kashmir", "Ladakh", "Puducherry", "Chandigarh",
    "Andaman & Nicobar Islands", "Dadra & Nagar Haveli",
    "Daman & Diu", "Lakshadweep"
]


def _list_items(items):
    rows = ""
    for i, item in enumerate(items, 1):
        rows += (f'<div class="list-item"><span class="list-num">{i}</span>'
                 f'<span class="list-name">{item}</span></div>')
    return rows


def _card_class(cid, decisions):
    d = decisions.get(cid)
    if d == 'keep':  return 'career-card-kept'
    if d == 'drop':  return 'career-card-dropped'
    return 'career-card'


def _status_pill(cid, decisions):
    d = decisions.get(cid)
    if d == 'keep':  return '<span class="status-pill-kept">✓ Kept</span>'
    if d == 'drop':  return '<span class="status-pill-dropped">✗ Dropped</span>'
    return '<span class="status-pill-none">Not decided yet</span>'


def _safe_secret(key, default=None):
    try:
        return st.secrets[key]
    except (KeyError, Exception):
        return default


def main():
    df, pinecone_needs_rebuild = load_careers()
    sentence_model = load_sentence_model()
    index          = load_pinecone_index()
    supabase       = load_supabase()
    svd_model      = load_svd_model(supabase)

    if pinecone_needs_rebuild and not st.session_state.get("pinecone_rebuilt"):
        from backend.onet_loader import rebuild_pinecone_after_refresh
        with st.spinner("🔄 Updating career search index..."):
            rebuild_pinecone_after_refresh(df, index, sentence_model)
        st.session_state["pinecone_rebuilt"] = True

    if df.empty:
        st.error("Career data could not be loaded. Check your data files and secrets.")
        return

    for key, default in [
        ('screen', 'profile'), ('profile', {}), ('riasec', {}),
        ('results', []),       ('selected_career', None),
        ('expanded_query', None), ('query_reviewed', False),
        ('query_was_expanded', False), ('show_expansion', False),
        ('decisions', {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown("""
    <div class="hero">
        <h1>🧭 CareerCompass India</h1>
        <p>Discover careers that match your stream, personality, and interests</p>
    </div>
    """, unsafe_allow_html=True)

    screens = ['profile', 'quiz', 'results', 'detail']
    steps   = ['📋 Profile', '🧭 RIASEC Quiz', '🎯 Recommendations', '🗺️ Career Detail']
    current = screens.index(st.session_state.screen) + 1
    cols = st.columns(4)
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 < current:    st.success(step)
            elif i + 1 == current: st.info(f"**{step}**")
            else: st.markdown(f"<div style='color:#aaa'>{step}</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # SCREEN 1 — PROFILE
    # ══════════════════════════════════════════════════════════
    if st.session_state.screen == 'profile':
        st.markdown("### 📋 Tell us about yourself")
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            with col1:
                name   = st.text_input("Your Name", placeholder="e.g. Riya Sharma")
                stream = st.selectbox("Class 12 Stream", ["Science","Commerce","Arts","Vocational"])
                marks  = st.slider("Your Marks (%)", 40, 100, 75)
                budget = st.selectbox("Annual Education Budget",
                            ["Under ₹50,000","₹50,000–₹1.5L","₹1.5L–₹5L","Above ₹5L"])
            with col2:
                city  = st.text_input("Your City", placeholder="e.g. Kurnool")
                state = st.selectbox("Your State", INDIAN_STATES)
                query = st.text_area("What are your interests?",
                    placeholder="e.g. I like biology  /  I enjoy maths  /  I love drawing",
                    height=120)
            submitted = st.form_submit_button("Check My Interests →")

        if submitted:
            if not name or not name.strip():
                st.error("Please enter your name.")
            elif not query or not query.strip():
                st.error("Please describe your interests.")
            else:
                st.session_state._temp_profile = {
                    "student_id": f"STU_{str(uuid.uuid4())[:8].upper()}",
                    "name": name.strip(), "stream": stream, "marks": marks,
                    "city": city, "state": state, "budget": budget,
                    "query": query.strip()
                }
                groq_key = _safe_secret("GROQ_API_KEY")
                with st.spinner("✨ Understanding your interests..."):
                    expanded, was_expanded = expand_query(query.strip(), groq_key)
                st.session_state._temp_expanded     = expanded
                st.session_state._temp_was_expanded = was_expanded
                st.session_state.show_expansion     = True
                st.rerun()

        if st.session_state.show_expansion and hasattr(st.session_state, '_temp_profile'):
            temp_profile = st.session_state._temp_profile
            expanded     = st.session_state._temp_expanded
            was_expanded = st.session_state._temp_was_expanded
            original     = temp_profile['query']

            if was_expanded:
                st.markdown("---")
                st.markdown("#### ✨ We understood your interests better!")
                st.markdown(
                    f'<div class="expand-box">'
                    f'<div class="expand-label">📝 What you wrote:</div>'
                    f'<div class="expand-original">{original}</div>'
                    f'<div class="expand-label">💡 What this means for your career search:</div>'
                    f'<div class="expand-result">{expanded}</div>'
                    f'</div>', unsafe_allow_html=True)
                st.caption("This helps find better career matches. You can use this or go with what you wrote.")
                st.markdown("")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Yes, use this — Continue to Quiz →", use_container_width=True):
                        st.session_state.profile            = temp_profile
                        st.session_state.expanded_query     = expanded
                        st.session_state.query_was_expanded = True
                        st.session_state.query_reviewed     = True
                        st.session_state.show_expansion     = False
                        st.session_state.results            = []
                        st.session_state.decisions          = {}
                        st.session_state.screen             = 'quiz'
                        st.rerun()
                with col_b:
                    if st.button("↩️ No, use my original — Continue to Quiz →", use_container_width=True):
                        st.session_state.profile            = temp_profile
                        st.session_state.expanded_query     = original
                        st.session_state.query_was_expanded = False
                        st.session_state.query_reviewed     = True
                        st.session_state.show_expansion     = False
                        st.session_state.results            = []
                        st.session_state.decisions          = {}
                        st.session_state.screen             = 'quiz'
                        st.rerun()
            else:
                st.session_state.profile            = temp_profile
                st.session_state.expanded_query     = original
                st.session_state.query_was_expanded = False
                st.session_state.query_reviewed     = True
                st.session_state.show_expansion     = False
                st.session_state.results            = []
                st.session_state.decisions          = {}
                st.session_state.screen             = 'quiz'
                st.rerun()

    # ══════════════════════════════════════════════════════════
    # SCREEN 2 — RIASEC QUIZ
    # ══════════════════════════════════════════════════════════
    elif st.session_state.screen == 'quiz':
        profile = st.session_state.profile
        st.markdown(f"### 🧭 Career Interest Quiz — {profile.get('name', '')}")
        st.markdown("Rate each activity from **1** (not interested) to **5** (very interested)")
        st.markdown("")
        with st.form("riasec_form"):
            answers = {}
            for i, q in enumerate(RIASEC_QUESTIONS):
                c1, c2 = st.columns([3,1])
                with c1: st.markdown(f"**Q{i+1}.** {q['q']}")
                with c2: answers[i] = st.select_slider(f"q{i}", options=[1,2,3,4,5], value=3, label_visibility="collapsed")
            if st.form_submit_button("Get My Recommendations →"):
                riasec = compute_riasec_scores(answers)
                st.session_state.riasec    = riasec
                st.session_state.screen    = 'results'
                st.session_state.results   = []
                st.session_state.decisions = {}
                st.rerun()
        if st.button("← Back to Profile"):
            st.session_state.screen = 'profile'
            st.session_state.show_expansion = False
            st.rerun()

    # ══════════════════════════════════════════════════════════
    # SCREEN 3 — RECOMMENDATIONS WITH KEEP / DROP
    # ══════════════════════════════════════════════════════════
    elif st.session_state.screen == 'results':
        profile   = st.session_state.profile
        riasec    = st.session_state.riasec
        decisions = st.session_state.decisions

        st.markdown("### 🎯 Your Career Recommendations")

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-box"><h3>{profile.get("stream","")}</h3><p>Your Stream</p></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-box"><h3>{riasec.get("riasec_code","")}</h3><p>RIASEC Code</p></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-box"><h3>{profile.get("marks","")}%</h3><p>Your Marks</p></div>', unsafe_allow_html=True)
        with c4:
            top_type = riasec.get("top2", ["R"])[0]
            st.markdown(f'<div class="metric-box"><h3>{RIASEC_LABELS.get(top_type, top_type)}</h3><p>Primary Personality</p></div>', unsafe_allow_html=True)
        st.markdown("")

        with st.expander("📊 View Your Full RIASEC Profile"):
            for rtype, score in riasec.get('ranked', []):
                st.markdown(f"**{RIASEC_LABELS.get(rtype, rtype)}** ({rtype}) — {score}/10")
                st.progress(int((score/10)*100)/100)
                st.caption(RIASEC_DESCRIPTIONS.get(rtype, ""))
        st.markdown("---")

        if not st.session_state.results:
            final_query = st.session_state.expanded_query or profile.get('query', '')
            if not final_query:
                st.error("No query found. Please go back to profile.")
                if st.button("← Back to Profile"):
                    st.session_state.screen = 'profile'
                    st.rerun()
                return

            with st.spinner("🔄 Finding your best career matches..."):
                # Cold-start is determined inside get_recommendations()
                # from the real interaction count for this student.
                # Do NOT pass is_cold_start here.
                st.session_state.results = get_recommendations(
                    user_id=profile['student_id'],
                    query=final_query,
                    student_stream=profile['stream'],
                    riasec_top2=riasec['top2'],
                    df=df,
                    sentence_model=sentence_model,
                    index=index,
                    svd_model=svd_model,
                    supabase=supabase,
                    top_k=10
                )

        results = st.session_state.results
        if not results:
            st.warning("No recommendations found. The search index may be empty.")
            if st.button("← Back to Profile"):
                st.session_state.screen = 'profile'
                st.rerun()
            return

        kept_count    = sum(1 for r in results if decisions.get(r['career_id']) == 'keep')
        dropped_count = sum(1 for r in results if decisions.get(r['career_id']) == 'drop')

        # Show cold-start or hybrid mode banner
        first_result = results[0] if results else {}
        if first_result.get('is_cold_start'):
            n_needed = 10 - first_result.get('n_real_interactions', 0)
            st.markdown(
                f'<div class="coldstart-banner">🔍 <b>Discovery Mode</b> — '
                f'Keep or drop careers below. After {n_needed} more decisions, '
                f'the system will activate personalised collaborative recommendations.</div>',
                unsafe_allow_html=True
            )
        else:
            cw = first_result.get('content_weight', 0.7)
            sw = first_result.get('collab_weight', 0.3)
            st.markdown(
                f'<div class="coldstart-banner">🤝 <b>Hybrid Mode</b> — '
                f'Content: {int(cw*100)}% · Collaborative: {int(sw*100)}% '
                f'(based on {first_result.get("n_real_interactions", 0)} of your past decisions)</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f"#### 🏆 Top {len(results)} Careers &nbsp;&nbsp;"
            f"<span style='font-size:13px;color:#22c55e'>✓ {kept_count} kept</span> &nbsp; "
            f"<span style='font-size:13px;color:#6b7280'>✗ {dropped_count} dropped</span>",
            unsafe_allow_html=True
        )
        st.caption("Keep careers you want to explore further. Drop ones that do not feel right.")
        st.markdown("")

        def _sort_key(r):
            d = decisions.get(r['career_id'], '')
            return 0 if d == 'keep' else (2 if d == 'drop' else 1)

        sorted_results = sorted(results, key=_sort_key)
        prev_section   = None

        for i, rec in enumerate(sorted_results):
            cid          = rec['career_id']
            decision     = decisions.get(cid, '')
            section      = _sort_key(rec)

            if section != prev_section:
                if section == 0:
                    st.markdown('<span class="section-header-kept">✓ Kept</span>', unsafe_allow_html=True)
                elif section == 1 and kept_count > 0:
                    st.markdown('<span class="section-header-neutral">Exploring</span>', unsafe_allow_html=True)
                elif section == 2:
                    st.markdown('<span class="section-header-dropped">✗ Dropped</span>', unsafe_allow_html=True)
                prev_section = section

            badge       = STREAM_BADGE.get(rec['stream'], 'badge-science')
            riasec_icon = ("⭐⭐" if rec.get('riasec_boost')==1.3 else "⭐" if rec.get('riasec_boost')==1.15 else "")
            stream_icon = "🎓" if rec.get('stream_boost', 1.0) > 1.0 else ""
            interest, personality, stream_exp = generate_explanation(
                rec, profile.get('query', ''), riasec['top2'], profile['stream']
            )

            original_rank = next((j+1 for j, r in enumerate(results) if r['career_id'] == cid), i+1)
            match_pct = int(rec.get('final_score', 0) * 100)

            st.markdown(f"""
            <div class="{_card_class(cid, decisions)}">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-size:1.1rem;font-weight:700">
                        #{original_rank} {rec['career']} {stream_icon}{riasec_icon}
                    </span>
                    <div style="text-align:right">
                        <span style="font-size:1.3rem;font-weight:700;color:#60a5fa">{match_pct}%</span><br>
                        <small style="color:#888">match score</small>
                    </div>
                </div>
                <div style="margin-top:0.5rem">
                    <span class="badge {badge}">{rec['stream']}</span>
                    <span class="badge" style="background:#2a3040;color:#aabbcc">{rec['sector']}</span>
                    <span class="badge" style="background:#1e2e40;color:#93c5fd">RIASEC: {rec['primary_riasec']}/{rec['secondary_riasec']}</span>
                </div>
                <div class="why-box">
                    💡 <b>Why this fits you:</b><br>
                    • {interest}<br>• {personality}<br>• {stream_exp}
                </div>
                <hr class="keep-drop-divider">
                {_status_pill(cid, decisions)}
            </div>
            """, unsafe_allow_html=True)

            btn1, btn2, btn3 = st.columns([1, 1, 2])

            with btn1:
                label = "✓ Kept — undo?" if decision == 'keep' else "✓ Keep it"
                if st.button(label, key=f"keep_{cid}", use_container_width=True):
                    if decision == 'keep':
                        st.session_state.decisions.pop(cid, None)
                    else:
                        st.session_state.decisions[cid] = 'keep'
                        save_interaction(supabase, profile['student_id'],
                                         cid, rec['career'], 5, profile['stream'])
                    st.rerun()

            with btn2:
                label = "✗ Dropped — undo?" if decision == 'drop' else "✗ Drop it"
                if st.button(label, key=f"drop_{cid}", use_container_width=True):
                    if decision == 'drop':
                        st.session_state.decisions.pop(cid, None)
                    else:
                        st.session_state.decisions[cid] = 'drop'
                        save_interaction(supabase, profile['student_id'],
                                         cid, rec['career'], 1, profile['stream'])
                    st.rerun()

            with btn3:
                if st.button("View Details →", key=f"detail_{cid}", use_container_width=True):
                    st.session_state.selected_career = rec
                    st.session_state.screen = 'detail'
                    st.rerun()

            st.markdown("")

        if st.button("← Start Over"):
            for key in ['screen','profile','riasec','results','selected_career',
                        'expanded_query','query_reviewed','query_was_expanded',
                        'show_expansion','decisions']:
                st.session_state.pop(key, None)
            st.rerun()

    # ══════════════════════════════════════════════════════════
    # SCREEN 4 — CAREER DETAIL (unchanged from original)
    # ══════════════════════════════════════════════════════════
    elif st.session_state.screen == 'detail':
        rec     = st.session_state.selected_career
        profile = st.session_state.profile
        if not rec:
            st.session_state.screen = 'results'
            st.rerun()
            return

        badge = STREAM_BADGE.get(rec.get('stream', ''), 'badge-science')
        state = profile.get('state', '')

        st.markdown(f"### 🗺️ {rec['career']}")
        st.markdown(f"""
        <span class="badge {badge}">{rec.get('stream','')}</span>
        <span class="badge" style="background:#2a3040;color:#aabbcc">{rec.get('sector','')}</span>
        <span class="badge" style="background:#1e2e40;color:#93c5fd">RIASEC: {rec.get('primary_riasec','')}/{rec.get('secondary_riasec','')}</span>
        """, unsafe_allow_html=True)
        st.markdown("")

        tab1, tab2 = st.tabs(["📊 Job Market", "🗺️ Career Pathway"])

        with tab1:
            serpapi_key  = _safe_secret("SERPAPI_KEY", "")
            groq_key     = _safe_secret("GROQ_API_KEY", "")
            news_api_key = _safe_secret("NEWS_API_KEY", None)

            with st.spinner("🌐 Fetching market data..."):
                try:
                    market = fetch_full_market_data(
                        career_title=rec['career'], sector=rec.get('sector',''),
                        stream=profile.get('stream',''), serpapi_key=serpapi_key,
                        groq_key=groq_key, news_api_key=news_api_key, supabase=supabase
                    )
                except Exception as e:
                    st.error(f"Could not load market data: {e}")
                    market = None

            if not market:
                st.info("Market data unavailable. Check your API keys in secrets.")
            else:
                future = market["future"]
                salary = future.get("salary", {})
                intel  = future.get("intelligence", {})
                comps  = future.get("companies", [])

                outlook     = intel.get("outlook", {})
                competition = intel.get("competition", {})
                policy      = intel.get("policy", {})
                trend       = outlook.get("trend", "Stable")
                trend_badge = TREND_BADGE.get(trend, "badge-stable")
                comp_level  = competition.get("level", "Moderate")
                comp_badge  = COMPETITION_BADGE.get(comp_level, "badge-moderate")

                gov_badge = '<span class="badge-pill badge-govbacked">Government backed</span>' \
                            if outlook.get("government_backed") else ""

                st.markdown(f"""
                <div class="mkt-card">
                    <p class="mkt-card-label">Career outlook</p>
                    <span class="badge-pill {trend_badge}">{trend}</span>{gov_badge}
                    <p class="explain-text">"{outlook.get('explanation','')}"</p>
                    <p class="source-text">{intel.get('source_outlook','') or ''}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<p class="mkt-card-label" style="margin-bottom:8px">WHAT YOU WILL EARN</p>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="sal-card"><p class="sal-label">Starting today</p><p class="sal-value">₹{salary.get("current_low","?")}L–₹{salary.get("current_high","?")}L/yr</p><p class="sal-explain">Current market average based on live job listings</p><p class="sal-source">{salary.get("salary_source","")}</p></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="sal-card highlight"><p class="sal-label">When you graduate ({salary.get("graduation_year","?")})</p><p class="sal-value blue">₹{salary.get("projected_low","?")}L–₹{salary.get("projected_high","?")}L/yr</p><p class="sal-explain">Based on ₹{salary.get("current_lpa","?")}L growing at {salary.get("growth_pct","?")}</p><p class="sal-source">{salary.get("growth_source","")}</p></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="sal-card"><p class="sal-label">5 years into career</p><p class="sal-value">₹{salary.get("mid_low","?")}L–₹{salary.get("mid_high","?")}L/yr</p><p class="sal-explain">Professionals with 5 years experience typically earn 2–2.5× starting salary</p></div>', unsafe_allow_html=True)

                comp_items = _list_items(comps) if comps else '<div class="list-item"><span class="list-name" style="color:#5a6a7a">No company data available</span></div>'
                st.markdown(f'<div class="mkt-card"><p class="mkt-card-label">🏢 Top Companies in India</p>{comp_items}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="mkt-card"><p class="mkt-card-label">How competitive is entry?</p><span class="badge-pill {comp_badge}">{comp_level}</span><p class="explain-text">"{competition.get("explanation","")}"</p></div>', unsafe_allow_html=True)

                if policy.get("exists") and policy.get("explanation"):
                    st.markdown(f'<div class="mkt-card"><p class="mkt-card-label">Government push</p><p class="explain-text">"{policy.get("explanation","")}"</p><p class="source-text">{intel.get("source_policy","") or ""}</p></div>', unsafe_allow_html=True)

                cache_label = "📦 Cached" if intel.get("from_cache") else "🔴 Live"
                st.caption(f"{cache_label} · Updated: {intel.get('last_updated','')} · Based on {intel.get('headlines_used', 0)} news sources")

        with tab2:
            groq_key = _safe_secret("GROQ_API_KEY", "")
            with st.spinner("🤖 Generating career roadmap..."):
                try:
                    pathway = fetch_career_pathway(
                        rec['career'], rec.get('stream',''),
                        rec.get('sector',''), groq_key, supabase
                    )
                except Exception as e:
                    st.error(f"Could not generate pathway: {e}")
                    pathway = None

            local_data = None
            if state and groq_key:
                with st.spinner(f"🔍 Loading {state} specific options..."):
                    try:
                        local_data = fetch_local_recommendations(
                            rec['career'], rec.get('stream',''),
                            state, groq_key, supabase
                        )
                    except Exception as e:
                        print(f"Local recommendations failed: {e}")
                        local_data = None

            if not pathway:
                st.warning("Could not generate pathway. Check your GROQ_API_KEY in secrets.")
            else:
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📚 Steps After Class 12**")
                    for step in pathway.get("after_class12", []):
                        st.markdown(f'<div class="step-box">→ {step}</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown("**📖 Courses to Take**")
                    for course in pathway.get("courses", []):
                        st.markdown(f'<div class="course-box"><b>{course.get("name","")}</b><br><small style="color:#8899aa">{course.get("duration","")} • {course.get("type","")}</small></div>', unsafe_allow_html=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**📝 National Entrance Exams**")
                    for exam in pathway.get("national_exams", []):
                        st.markdown(f'<div class="exam-box"><b>{exam.get("exam","")}</b><br><small style="color:#8899aa">By {exam.get("conducted_by","")} • {exam.get("frequency","")}</small></div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f"**📝 State Entrance Exams — {state or 'Your State'}**")
                    if local_data and local_data.get("state_exams"):
                        for exam in local_data["state_exams"]:
                            st.markdown(f'<div class="exam-box" style="border-left-color:#f97316"><b>{exam.get("exam","")}</b><br><small style="color:#8899aa">By {exam.get("conducted_by","")} • {exam.get("eligibility","")}</small></div>', unsafe_allow_html=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                col5, col6 = st.columns(2)
                with col5:
                    st.markdown("**🎓 Top National Colleges**")
                    for college in pathway.get("top_colleges", []):
                        st.markdown(f'<div class="step-box">🏫 <b>{college.get("name","")}</b> — {college.get("city","")}</div>', unsafe_allow_html=True)
                with col6:
                    st.markdown(f"**🏫 Colleges in {state or 'Your State'}**")
                    if local_data and local_data.get("state_colleges"):
                        for college in local_data["state_colleges"]:
                            st.markdown(f'<div class="step-box" style="border-left-color:#86efac"><b>{college.get("name","")}</b><br><small style="color:#8899aa">{college.get("city","")} • {college.get("type","")}</small></div>', unsafe_allow_html=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("**💡 Skills to Develop**")
                skills = pathway.get("skills_to_develop", [])
                if not skills and rec.get('core_skills'):
                    skills = [{"skill": s.strip(), "level": ""} for s in rec['core_skills'].split(',') if s.strip()][:6]
                if skills:
                    skill_cols = st.columns(3)
                    for idx, skill in enumerate(skills):
                        with skill_cols[idx % 3]:
                            name  = skill.get('skill','') if isinstance(skill, dict) else str(skill)
                            level = skill.get('level','') if isinstance(skill, dict) else ''
                            level_html = f'<br><small style="color:#8899aa">{level}</small>' if level else ''
                            st.markdown(f'<div class="skill-box"><b>{name}</b>{level_html}</div>', unsafe_allow_html=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                col7, col8 = st.columns(2)
                with col7:
                    st.markdown("**🎓 National Scholarships**")
                    for s in pathway.get("national_scholarships", []):
                        st.markdown(f'<div class="scholarship-box"><b>{s.get("name","")}</b><br><small style="color:#8899aa">💰 {s.get("amount","")} • {s.get("eligibility","")}</small></div>', unsafe_allow_html=True)
                with col8:
                    st.markdown(f"**🎓 Scholarships in {state or 'Your State'}**")
                    if local_data and local_data.get("state_scholarships"):
                        for s in local_data["state_scholarships"]:
                            st.markdown(f'<div class="scholarship-box" style="border-left-color:#fcd34d"><b>{s.get("name","")}</b><br><small style="color:#8899aa">💰 {s.get("amount","")} • {s.get("eligibility","")}</small></div>', unsafe_allow_html=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                col9, col10 = st.columns(2)
                with col9:
                    st.markdown("**📅 Year-by-Year Timeline**")
                    for t in pathway.get("timeline", []):
                        st.markdown(f'<div class="step-box"><b>{t.get("year","")}</b> → {t.get("milestone","")}</div>', unsafe_allow_html=True)
                with col10:
                    st.markdown("**📈 Career Progression**")
                    for level in pathway.get("career_progression", []):
                        st.markdown(f'<div class="step-box">→ {level}</div>', unsafe_allow_html=True)
                    avg_sal = pathway.get("avg_starting_salary", "")
                    if avg_sal and avg_sal != "Not available":
                        st.markdown(f'<div class="metric-box" style="margin-top:1rem"><h3 style="font-size:1.1rem">💰 {avg_sal}</h3><p>Average Starting Salary</p></div>', unsafe_allow_html=True)

                st.caption("⚠️ AI-generated roadmap — verify exam and college details at official websites.")

        st.markdown("")
        if st.button("← Back to Recommendations"):
            st.session_state.screen = 'results'
            st.rerun()


if __name__ == "__main__":
    main()
