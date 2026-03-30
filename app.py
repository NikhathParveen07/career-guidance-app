# ============================================
# app.py — UI only
# All logic lives in backend/
# ============================================
import streamlit as st
import uuid

from backend.data_loader    import load_careers, load_sentence_model, load_pinecone_index, load_supabase
from backend.collaborative  import load_svd_model
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
        margin-bottom: 0.8rem; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border-left: 4px solid #e94560; color: #f0f0f0;
    }
    .career-card span  { color: #f0f0f0 !important; }
    .career-card small { color: #aaaaaa !important; }

    .expand-box {
        background: #1a2a1a; border-radius: 10px; padding: 1rem 1.2rem;
        border: 1px solid #2a5a2a; margin: 0.8rem 0;
        font-size: 0.9rem; color: #c8e6c9; line-height: 1.6;
    }
    .expand-label {
        font-size: 11px; color: #66bb6a; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 6px; font-weight: 600;
    }

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

    .pathway-card {
        background: #1e2530; border-radius: 10px; padding: 1rem 1.2rem;
        border: 1px solid #2e3a4e; margin-bottom: 0.6rem;
    }
    .pathway-section-title {
        font-size: 11px; color: #8899aa; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 8px; font-weight: 600;
    }

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

    .step-box {
        background: #1e2530; border-radius: 8px; padding: 0.8rem 1rem;
        margin: 0.4rem 0; border-left: 3px solid #60a5fa;
        color: #d0d8e8; line-height: 1.5;
    }
    .step-box b     { color: #93c5fd; }
    .step-box small { color: #8899aa; }

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
        rows += f"""
        <div class="list-item">
            <span class="list-num">{i}</span>
            <span class="list-name">{item}</span>
        </div>"""
    return rows


def main():
    df             = load_careers()
    sentence_model = load_sentence_model()
    index          = load_pinecone_index()
    supabase       = load_supabase()
    svd_model      = load_svd_model()

    for key, default in [
        ('screen', 'profile'), ('profile', {}), ('riasec', {}),
        ('results', []),       ('selected_career', None),
        ('expanded_query', None), ('query_reviewed', False)
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
            if i + 1 < current:       st.success(step)
            elif i + 1 == current:    st.info(f"**{step}**")
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
                            placeholder="e.g. I love biology, drawing, and helping people",
                            height=120)
            if st.form_submit_button("Continue to RIASEC Quiz →"):
                if not name or not query:
                    st.error("Please fill in your name and interests.")
                else:
                    st.session_state.profile = {
                        "student_id": f"STU_{str(uuid.uuid4())[:8].upper()}",
                        "name": name, "stream": stream, "marks": marks,
                        "city": city, "state": state, "budget": budget, "query": query
                    }
                    st.session_state.expanded_query = None
                    st.session_state.query_reviewed = False
                    st.session_state.screen = 'quiz'
                    st.rerun()

    # ══════════════════════════════════════════════════════════
    # SCREEN 2 — RIASEC QUIZ
    # ══════════════════════════════════════════════════════════
    elif st.session_state.screen == 'quiz':
        profile = st.session_state.profile
        st.markdown(f"### 🧭 Career Interest Quiz — {profile['name']}")
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
                st.session_state.riasec  = riasec
                st.session_state.screen  = 'results'
                st.session_state.results = []
                st.rerun()
        if st.button("← Back to Profile"):
            st.session_state.screen = 'profile'; st.rerun()

    # ══════════════════════════════════════════════════════════
    # SCREEN 3 — RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════
    elif st.session_state.screen == 'results':
        profile = st.session_state.profile
        riasec  = st.session_state.riasec
        st.markdown("### 🎯 Your Career Recommendations")

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-box"><h3>{profile["stream"]}</h3><p>Your Stream</p></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-box"><h3>{riasec["riasec_code"]}</h3><p>RIASEC Code</p></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-box"><h3>{profile["marks"]}%</h3><p>Your Marks</p></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-box"><h3>{RIASEC_LABELS[riasec["top2"][0]]}</h3><p>Primary Personality</p></div>', unsafe_allow_html=True)
        st.markdown("")

        with st.expander("📊 View Your Full RIASEC Profile"):
            for rtype, score in riasec['ranked']:
                st.markdown(f"**{RIASEC_LABELS[rtype]}** ({rtype}) — {score}/10")
                st.progress(int((score/10)*100)/100)
                st.caption(RIASEC_DESCRIPTIONS[rtype])
        st.markdown("---")

        # ── Task 3: Query Expansion UI ────────────────────────
        if not st.session_state.results:

            original_query = profile['query']

            # Run expansion if not done yet
            if st.session_state.expanded_query is None:
                with st.spinner("🔍 Understanding your interests..."):
                    expanded, was_expanded = expand_query(
                        original_query,
                        st.secrets["GROQ_API_KEY"]
                    )
                    st.session_state.expanded_query  = expanded
                    st.session_state.query_was_expanded = was_expanded

            expanded_query  = st.session_state.expanded_query
            was_expanded    = st.session_state.get('query_was_expanded', False)
            query_reviewed  = st.session_state.query_reviewed

            # Show expansion review UI only if query was expanded and not yet reviewed
            if was_expanded and not query_reviewed:
                st.markdown("#### 🤖 We enriched your interest description")
                st.markdown(
                    f'<div class="expand-box">'
                    f'<div class="expand-label">✨ Your original: </div>'
                    f'<i style="color:#aaa">{original_query}</i>'
                    f'<br><br>'
                    f'<div class="expand-label">🚀 Enriched version: </div>'
                    f'{expanded_query}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.caption(
                    "We expanded your description to help find better career matches. "
                    "You can accept this enriched version or use your original."
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Use enriched version", use_container_width=True):
                        st.session_state.query_reviewed = True
                        st.session_state.results = []
                        st.rerun()
                with col_b:
                    if st.button("↩️ Use my original", use_container_width=True):
                        st.session_state.expanded_query  = original_query
                        st.session_state.query_reviewed  = True
                        st.session_state.results = []
                        st.rerun()

            else:
                # Query confirmed — run recommendations
                with st.spinner("🔄 Finding your best career matches..."):
                    final_query = st.session_state.expanded_query or original_query
                    st.session_state.results = get_recommendations(
                        user_id=profile['student_id'], query=final_query,
                        student_stream=profile['stream'], riasec_top2=riasec['top2'],
                        df=df, sentence_model=sentence_model, index=index,
                        svd_model=svd_model, is_cold_start=True, top_k=10
                    )
                st.rerun()

        results = st.session_state.results
        if not results:
            st.info("Processing your interests...")
            return

        # Show which query was used
        final_query = st.session_state.expanded_query or profile['query']
        if st.session_state.get('query_was_expanded') and final_query != profile['query']:
            st.caption(f"🔍 Recommendations based on enriched interests")

        st.markdown(f"#### 🏆 Top {len(results)} Careers for You")

        for i, rec in enumerate(results):
            badge       = STREAM_BADGE.get(rec['stream'], 'badge-science')
            riasec_icon = ("⭐⭐" if rec['riasec_boost']==1.3 else "⭐" if rec['riasec_boost']==1.15 else "")
            stream_icon = "🎓" if rec['stream_boost']>1.0 else ""
            interest, personality, stream_exp = generate_explanation(rec, profile['query'], riasec['top2'], profile['stream'])

            st.markdown(f"""
            <div class="career-card">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-size:1.1rem;font-weight:700;color:#f0f0f0">
                        #{i+1} {rec['career']} {stream_icon}{riasec_icon}
                    </span>
                    <div style="text-align:right">
                        <span style="font-size:1.3rem;font-weight:700;color:#60a5fa">{int(rec['final_score']*100)}%</span><br>
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
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"View Details →", key=f"detail_{i}"):
                st.session_state.selected_career = rec
                st.session_state.screen = 'detail'
                st.rerun()

        if st.button("← Start Over"):
            for key in ['screen','profile','riasec','results','selected_career',
                        'expanded_query','query_reviewed','query_was_expanded']:
                st.session_state.pop(key, None)
            st.rerun()

    # ══════════════════════════════════════════════════════════
    # SCREEN 4 — CAREER DETAIL
    # ══════════════════════════════════════════════════════════
    elif st.session_state.screen == 'detail':
        rec     = st.session_state.selected_career
        profile = st.session_state.profile
        if not rec:
            st.session_state.screen = 'results'; st.rerun()

        badge = STREAM_BADGE.get(rec['stream'], 'badge-science')
        state = profile.get('state', '')

        st.markdown(f"### 🗺️ {rec['career']}")
        st.markdown(f"""
        <span class="badge {badge}">{rec['stream']}</span>
        <span class="badge" style="background:#2a3040;color:#aabbcc">{rec['sector']}</span>
        <span class="badge" style="background:#1e2e40;color:#93c5fd">RIASEC: {rec['primary_riasec']}/{rec['secondary_riasec']}</span>
        """, unsafe_allow_html=True)
        st.markdown("")

        tab1, tab2 = st.tabs(["📊 Job Market", "🗺️ Career Pathway"])

        # ══════════════════════════════════════════════════════
        # TAB 1 — JOB MARKET
        # ══════════════════════════════════════════════════════
        with tab1:
            with st.spinner("🌐 Fetching market data..."):
                market = fetch_full_market_data(
                    career_title = rec['career'],
                    sector       = rec['sector'],
                    stream       = profile['stream'],
                    serpapi_key  = st.secrets["SERPAPI_KEY"],
                    groq_key     = st.secrets["GROQ_API_KEY"],
                    news_api_key = st.secrets.get("NEWS_API_KEY", None),
                    supabase     = supabase
                )

            future = market["future"]
            salary = future["salary"]
            intel  = future["intelligence"]
            cities = future["cities"]
            comps  = future["companies"]

            outlook     = intel.get("outlook", {})
            competition = intel.get("competition", {})
            policy      = intel.get("policy", {})

            trend       = outlook.get("trend", "Stable")
            trend_badge = TREND_BADGE.get(trend, "badge-stable")
            comp_level  = competition.get("level", "Moderate")
            comp_badge  = COMPETITION_BADGE.get(comp_level, "badge-moderate")

            grad_year  = salary.get("graduation_year", "")
            curr_low   = salary.get("current_low", "")
            curr_high  = salary.get("current_high", "")
            proj_low   = salary.get("projected_low", "")
            proj_high  = salary.get("projected_high", "")
            mid_low    = salary.get("mid_low", "")
            mid_high   = salary.get("mid_high", "")
            growth_pct = salary.get("growth_pct", "")
            sal_src    = salary.get("salary_source", "")
            grow_src   = salary.get("growth_source", "")
            curr_lpa   = salary.get("current_lpa", "")

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

            st.markdown(
                '<p class="mkt-card-label" style="margin-bottom:8px">WHAT YOU WILL EARN</p>',
                unsafe_allow_html=True
            )
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class="sal-card">
                    <p class="sal-label">Starting today</p>
                    <p class="sal-value">₹{curr_low}L–₹{curr_high}L/yr</p>
                    <p class="sal-explain">Current market average based on live job listings</p>
                    <p class="sal-source">{sal_src}</p>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="sal-card highlight">
                    <p class="sal-label">When you graduate ({grad_year})</p>
                    <p class="sal-value blue">₹{proj_low}L–₹{proj_high}L/yr</p>
                    <p class="sal-explain">Based on current ₹{curr_lpa}L growing at {growth_pct}</p>
                    <p class="sal-source">{grow_src}</p>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="sal-card">
                    <p class="sal-label">5 years into career</p>
                    <p class="sal-value">₹{mid_low}L–₹{mid_high}L/yr</p>
                    <p class="sal-explain">Professionals with 5 years experience earn 2–2.5x starting salary</p>
                </div>
                """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                city_items = _list_items(cities) if cities else \
                    '<div class="list-item"><span class="list-name" style="color:#5a6a7a">No location data available</span></div>'
                st.markdown(f"""
                <div class="mkt-card">
                    <p class="mkt-card-label">Where the jobs are</p>
                    {city_items}
                </div>
                """, unsafe_allow_html=True)

            with col2:
                comp_items = _list_items(comps) if comps else \
                    '<div class="list-item"><span class="list-name" style="color:#5a6a7a">No company data available</span></div>'
                st.markdown(f"""
                <div class="mkt-card">
                    <p class="mkt-card-label">Who will hire you</p>
                    {comp_items}
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="mkt-card">
                <p class="mkt-card-label">How competitive is entry?</p>
                <span class="badge-pill {comp_badge}">{comp_level}</span>
                <p class="explain-text">"{competition.get('explanation','')}"</p>
            </div>
            """, unsafe_allow_html=True)

            if policy.get("exists") and policy.get("explanation"):
                st.markdown(f"""
                <div class="mkt-card">
                    <p class="mkt-card-label">Government push</p>
                    <p class="explain-text">"{policy.get('explanation','')}"</p>
                    <p class="source-text">{intel.get('source_policy','') or ''}</p>
                </div>
                """, unsafe_allow_html=True)

            cache_label = "📦 Cached" if intel.get("from_cache") else "🔴 Live"
            st.caption(
                f"{cache_label} · Updated: {intel.get('last_updated','')} · "
                f"Based on {intel.get('headlines_used', 0)} news sources"
            )

        # ══════════════════════════════════════════════════════
        # TAB 2 — CAREER PATHWAY (Task 4 — redesigned)
        # ══════════════════════════════════════════════════════
        with tab2:
            with st.spinner("🤖 Generating career roadmap..."):
                pathway = fetch_career_pathway(
                    rec['career'], rec['stream'], rec['sector'],
                    st.secrets["GROQ_API_KEY"], supabase
                )
            local_data = None
            if state:
                with st.spinner(f"🔍 Loading {state} specific options..."):
                    local_data = fetch_local_recommendations(
                        rec['career'], rec['stream'], state,
                        st.secrets["GROQ_API_KEY"], supabase
                    )

            if not pathway:
                st.warning("Could not generate pathway. Please try again.")
            else:
                # ── Row 1: Steps + Courses ────────────────────
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📚 Steps After Class 12**")
                    for step in pathway.get("after_class12", []):
                        st.markdown(
                            f'<div class="step-box">→ {step}</div>',
                            unsafe_allow_html=True)

                with col2:
                    st.markdown("**📖 Courses to Take**")
                    courses = pathway.get("courses", [])
                    if courses:
                        for course in courses:
                            name     = course.get('name', '')
                            duration = course.get('duration', '')
                            ctype    = course.get('type', '')
                            st.markdown(f"""
                            <div class="course-box">
                                <b>{name}</b><br>
                                <small style="color:#8899aa">{duration} • {ctype}</small>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.info("Course data not available.")

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # ── Row 2: National Exams + State Exams ──────
                col3, col4 = st.columns(2)

                with col3:
                    st.markdown("**📝 National Entrance Exams**")
                    for exam in pathway.get("national_exams", []):
                        st.markdown(f"""
                        <div class="exam-box">
                            <b>{exam.get('exam','')}</b><br>
                            <small style="color:#8899aa">
                                By {exam.get('conducted_by','')} &nbsp;•&nbsp; {exam.get('frequency','')}
                            </small>
                        </div>""", unsafe_allow_html=True)

                with col4:
                    st.markdown(f"**📝 State Entrance Exams — {state or 'Your State'}**")
                    if local_data and local_data.get("state_exams"):
                        for exam in local_data.get("state_exams", []):
                            st.markdown(f"""
                            <div class="exam-box" style="border-left-color:#f97316">
                                <b>{exam.get('exam','')}</b><br>
                                <small style="color:#8899aa">
                                    By {exam.get('conducted_by','')} &nbsp;•&nbsp; {exam.get('eligibility','')}
                                </small>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.info("Select your state on the profile page for state-specific exams.")

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # ── Row 3: Top Colleges + State Colleges ─────
                col5, col6 = st.columns(2)

                with col5:
                    st.markdown("**🎓 Top National Colleges**")
                    for college in pathway.get("top_colleges", []):
                        st.markdown(
                            f'<div class="step-box">🏫 <b>{college.get("name","")}</b> — {college.get("city","")}</div>',
                            unsafe_allow_html=True)

                with col6:
                    st.markdown(f"**🏫 Colleges in {state or 'Your State'}**")
                    if local_data and local_data.get("state_colleges"):
                        for college in local_data.get("state_colleges", []):
                            ctype = college.get('type','')
                            st.markdown(f"""
                            <div class="step-box" style="border-left-color:#86efac">
                                <b>{college.get('name','')}</b><br>
                                <small style="color:#8899aa">{college.get('city','')} • <span style="color:#86efac">{ctype}</span></small>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.info("Select your state on the profile page for state colleges.")

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # ── Row 4: Skills to Develop ──────────────────
                st.markdown("**💡 Skills to Develop**")
                skills = pathway.get("skills_to_develop", [])
                if skills:
                    skill_cols = st.columns(3)
                    for i, skill in enumerate(skills):
                        with skill_cols[i % 3]:
                            name  = skill.get('skill', skill) if isinstance(skill, dict) else skill
                            level = skill.get('level', '') if isinstance(skill, dict) else ''
                            st.markdown(f"""
                            <div class="skill-box">
                                <b>{name}</b>
                                {"<br><small style='color:#8899aa'>" + level + "</small>" if level else ""}
                            </div>""", unsafe_allow_html=True)
                else:
                    # Fallback to core_skills from O*NET
                    if rec.get('core_skills'):
                        skills_list = [s.strip() for s in rec['core_skills'].split(',') if s.strip()]
                        skill_cols  = st.columns(3)
                        for i, skill in enumerate(skills_list[:6]):
                            with skill_cols[i % 3]:
                                st.markdown(
                                    f'<div class="skill-box"><b>{skill}</b></div>',
                                    unsafe_allow_html=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # ── Row 5: Scholarships ───────────────────────
                col7, col8 = st.columns(2)

                with col7:
                    st.markdown("**🎓 National Scholarships**")
                    national_scholarships = pathway.get("national_scholarships", [])
                    if national_scholarships:
                        for s in national_scholarships:
                            st.markdown(f"""
                            <div class="scholarship-box">
                                <b>{s.get('name','')}</b><br>
                                <small style="color:#8899aa">
                                    💰 {s.get('amount','')} &nbsp;•&nbsp; {s.get('eligibility','')}
                                </small>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.info("No national scholarship data available.")

                with col8:
                    st.markdown(f"**🎓 Scholarships in {state or 'Your State'}**")
                    if local_data and local_data.get("state_scholarships"):
                        for s in local_data.get("state_scholarships", []):
                            st.markdown(f"""
                            <div class="scholarship-box" style="border-left-color:#fcd34d">
                                <b>{s.get('name','')}</b><br>
                                <small style="color:#8899aa">
                                    💰 {s.get('amount','')} &nbsp;•&nbsp; {s.get('eligibility','')}
                                </small>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.info("Select your state on the profile page for state scholarships.")

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # ── Row 6: Timeline + Career Progression ─────
                col9, col10 = st.columns(2)

                with col9:
                    st.markdown("**📅 Year-by-Year Timeline**")
                    for t in pathway.get("timeline", []):
                        st.markdown(
                            f'<div class="step-box"><b>{t.get("year","")}</b> → {t.get("milestone","")}</div>',
                            unsafe_allow_html=True)

                with col10:
                    st.markdown("**📈 Career Progression**")
                    for level in pathway.get("career_progression", []):
                        st.markdown(
                            f'<div class="step-box">→ {level}</div>',
                            unsafe_allow_html=True)
                    st.markdown("")
                    sal = pathway.get("avg_starting_salary","")
                    if sal:
                        st.markdown(
                            f'<div class="metric-box"><h3 style="font-size:1.1rem">💰 {sal}</h3><p>Average Starting Salary</p></div>',
                            unsafe_allow_html=True)

                st.caption("⚠️ AI-generated roadmap — verify exam and college details at official websites.")

        st.markdown("")
        if st.button("← Back to Recommendations"):
            st.session_state.screen = 'results'; st.rerun()


if __name__ == "__main__":
    main()
