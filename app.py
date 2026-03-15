# ============================================
# app.py
# UI only — screens, layout, CSS
# All logic lives in backend/
# ============================================
import streamlit as st
import uuid

from backend.data_loader    import load_careers, load_sentence_model, load_pinecone_index, load_supabase
from backend.collaborative  import load_svd_model
from backend.hybrid_engine  import get_recommendations
from backend.riasec         import RIASEC_QUESTIONS, RIASEC_LABELS, RIASEC_DESCRIPTIONS, compute_riasec_scores
from backend.explainability import generate_explanation
from backend.job_market     import fetch_job_market_data
from backend.pathway        import fetch_career_pathway, fetch_local_recommendations


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title            = "CareerCompass India",
    page_icon             = "🧭",
    layout                = "wide",
    initial_sidebar_state = "collapsed"
)

# ── CSS ───────────────────────────────────────────────────────
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
        background: #1e2530;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border-left: 4px solid #e94560;
        color: #f0f0f0;
    }
    .career-card span  { color: #f0f0f0 !important; }
    .career-card small { color: #aaaaaa !important; }

    .metric-box {
        background: #1e2530;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2e3a4e;
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
        background: #2a2510;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        border: 1px solid #5a4a20;
        font-size: 0.85rem;
        color: #e5d5a0;
        line-height: 1.6;
    }
    .why-box b { color: #fcd34d; }

    .step-box {
        background: #1e2530;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border-left: 3px solid #60a5fa;
        color: #d0d8e8;
        line-height: 1.5;
    }
    .step-box b     { color: #93c5fd; }
    .step-box small { color: #8899aa; }

    .section-divider {
        border: none;
        border-top: 1px solid #2e3a4e;
        margin: 1.5rem 0;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #0f3460, #e94560);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 2rem; font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ── Stream badge colours ──────────────────────────────────────
STREAM_BADGE = {
    "Science":    "badge-science",
    "Commerce":   "badge-commerce",
    "Arts":       "badge-arts",
    "Vocational": "badge-vocational"
}

# ── Indian states list ────────────────────────────────────────
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu & Kashmir", "Ladakh",
    "Puducherry", "Chandigarh",
    "Andaman & Nicobar Islands", "Dadra & Nagar Haveli",
    "Daman & Diu", "Lakshadweep"
]


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():

    df             = load_careers()
    sentence_model = load_sentence_model()
    index          = load_pinecone_index()
    supabase       = load_supabase()
    svd_model      = load_svd_model()

    for key, default in [
        ('screen',          'profile'),
        ('profile',         {}),
        ('riasec',          {}),
        ('results',         []),
        ('selected_career', None)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Hero ──────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🧭 CareerCompass India</h1>
        <p>Discover careers that match your stream, personality, and interests</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Progress indicator ────────────────────────────────────
    screens = ['profile', 'quiz', 'results', 'detail']
    steps   = ['📋 Profile', '🧭 RIASEC Quiz', '🎯 Recommendations', '🗺️ Career Detail']
    current = screens.index(st.session_state.screen) + 1

    cols = st.columns(4)
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 < current:
                st.success(step)
            elif i + 1 == current:
                st.info(f"**{step}**")
            else:
                st.markdown(f"<div style='color:#aaa'>{step}</div>",
                            unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # SCREEN 1 — STUDENT PROFILE
    # ══════════════════════════════════════════════════════════
    if st.session_state.screen == 'profile':

        st.markdown("### 📋 Tell us about yourself")

        with st.form("profile_form"):
            col1, col2 = st.columns(2)

            with col1:
                name   = st.text_input("Your Name",
                            placeholder="e.g. Riya Sharma")
                stream = st.selectbox("Class 12 Stream",
                            ["Science", "Commerce", "Arts", "Vocational"])
                marks  = st.slider("Your Marks (%)", 40, 100, 75)
                budget = st.selectbox("Annual Education Budget",
                            ["Under ₹50,000", "₹50,000–₹1.5L",
                             "₹1.5L–₹5L",    "Above ₹5L"])

            with col2:
                city  = st.text_input("Your City",
                            placeholder="e.g. Kurnool")
                state = st.selectbox("Your State", INDIAN_STATES)
                query = st.text_area("What are your interests?",
                            placeholder="e.g. I love biology, drawing, and helping people",
                            height=120)

            submitted = st.form_submit_button("Continue to RIASEC Quiz →")

            if submitted:
                if not name or not query:
                    st.error("Please fill in your name and interests.")
                else:
                    st.session_state.profile = {
                        "student_id": f"STU_{str(uuid.uuid4())[:8].upper()}",
                        "name":       name,
                        "stream":     stream,
                        "marks":      marks,
                        "city":       city,
                        "state":      state,
                        "budget":     budget,
                        "query":      query
                    }
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
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Q{i+1}.** {q['q']}")
                with c2:
                    answers[i] = st.select_slider(
                        f"q{i}", options=[1,2,3,4,5],
                        value=3, label_visibility="collapsed"
                    )

            submitted = st.form_submit_button("Get My Recommendations →")

            if submitted:
                riasec = compute_riasec_scores(answers)
                st.session_state.riasec  = riasec
                st.session_state.screen  = 'results'
                st.session_state.results = []
                st.rerun()

        if st.button("← Back to Profile"):
            st.session_state.screen = 'profile'
            st.rerun()

    # ══════════════════════════════════════════════════════════
    # SCREEN 3 — RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════
    elif st.session_state.screen == 'results':

        profile = st.session_state.profile
        riasec  = st.session_state.riasec

        st.markdown("### 🎯 Your Career Recommendations")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-box"><h3>{profile["stream"]}</h3><p>Your Stream</p></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="metric-box"><h3>{riasec["riasec_code"]}</h3><p>RIASEC Code</p></div>',
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                f'<div class="metric-box"><h3>{profile["marks"]}%</h3><p>Your Marks</p></div>',
                unsafe_allow_html=True)
        with c4:
            st.markdown(
                f'<div class="metric-box"><h3>{RIASEC_LABELS[riasec["top2"][0]]}</h3><p>Primary Personality</p></div>',
                unsafe_allow_html=True)

        st.markdown("")

        with st.expander("📊 View Your Full RIASEC Profile"):
            for rtype, score in riasec['ranked']:
                st.markdown(f"**{RIASEC_LABELS[rtype]}** ({rtype}) — {score}/10")
                st.progress(int((score / 10) * 100) / 100)
                st.caption(RIASEC_DESCRIPTIONS[rtype])

        st.markdown("---")

        if not st.session_state.results:
            with st.spinner("🔄 Finding your best career matches..."):
                recs = get_recommendations(
                    user_id        = profile['student_id'],
                    query          = profile['query'],
                    student_stream = profile['stream'],
                    riasec_top2    = riasec['top2'],
                    df             = df,
                    sentence_model = sentence_model,
                    index          = index,
                    svd_model      = svd_model,
                    is_cold_start  = True,
                    top_k          = 10
                )
                st.session_state.results = recs

        results = st.session_state.results
        st.markdown(f"#### 🏆 Top {len(results)} Careers for You")

        for i, rec in enumerate(results):
            badge       = STREAM_BADGE.get(rec['stream'], 'badge-science')
            riasec_icon = ("⭐⭐" if rec['riasec_boost'] == 1.3  else
                           "⭐"   if rec['riasec_boost'] == 1.15 else "")
            stream_icon = "🎓" if rec['stream_boost'] > 1.0 else ""

            interest, personality, stream_exp = generate_explanation(
                rec, profile['query'], riasec['top2'], profile['stream']
            )

            st.markdown(f"""
            <div class="career-card">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-size:1.1rem;font-weight:700;color:#f0f0f0">
                        #{i+1} {rec['career']} {stream_icon}{riasec_icon}
                    </span>
                    <div style="text-align:right">
                        <span style="font-size:1.3rem;font-weight:700;color:#60a5fa">
                            {int(rec['final_score']*100)}%
                        </span><br>
                        <small style="color:#888888">match score</small>
                    </div>
                </div>
                <div style="margin-top:0.5rem">
                    <span class="badge {badge}">{rec['stream']}</span>
                    <span class="badge" style="background:#2a3040;color:#aabbcc">{rec['sector']}</span>
                    <span class="badge" style="background:#1e2e40;color:#93c5fd">
                        RIASEC: {rec['primary_riasec']}/{rec['secondary_riasec']}
                    </span>
                </div>
                <div class="why-box">
                    💡 <b>Why this fits you:</b><br>
                    • {interest}<br>
                    • {personality}<br>
                    • {stream_exp}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"View Details →", key=f"detail_{i}"):
                st.session_state.selected_career = rec
                st.session_state.screen = 'detail'
                st.rerun()

        if st.button("← Start Over"):
            for key in ['screen','profile','riasec','results','selected_career']:
                st.session_state.pop(key, None)
            st.rerun()

    # ══════════════════════════════════════════════════════════
    # SCREEN 4 — CAREER DETAIL
    # ══════════════════════════════════════════════════════════
    elif st.session_state.screen == 'detail':

        rec     = st.session_state.selected_career
        profile = st.session_state.profile

        if not rec:
            st.session_state.screen = 'results'
            st.rerun()

        badge = STREAM_BADGE.get(rec['stream'], 'badge-science')
        state = profile.get('state', '')

        st.markdown(f"### 🗺️ {rec['career']}")
        st.markdown(f"""
        <span class="badge {badge}">{rec['stream']}</span>
        <span class="badge" style="background:#2a3040;color:#aabbcc">{rec['sector']}</span>
        <span class="badge" style="background:#1e2e40;color:#93c5fd">
            RIASEC: {rec['primary_riasec']}/{rec['secondary_riasec']}
        </span>
        """, unsafe_allow_html=True)
        st.markdown("")

        tab1, tab2 = st.tabs(["📊 Job Market", "🗺️ Career Pathway"])

        # ══════════════════════════════════════════════════════
        # TAB 1 — JOB MARKET
        # ══════════════════════════════════════════════════════
        with tab1:
            with st.spinner("🌐 Fetching live job data..."):
                job = fetch_job_market_data(
                    rec['career'], st.secrets["SERPAPI_KEY"]
                )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="metric-box"><h3>{job["total"]}+</h3><p>Sample Openings</p></div>',
                    unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f'<div class="metric-box"><h3 style="font-size:1rem">{job["demand"]}</h3><p>Market Demand</p></div>',
                    unsafe_allow_html=True)
            with c3:
                st.markdown(
                    f'<div class="metric-box"><h3 style="font-size:1rem">{job["salary"]}</h3><p>Salary Range</p></div>',
                    unsafe_allow_html=True)

            if job['companies']:
                st.markdown("")
                st.markdown("**🏢 Top Hiring Companies:**")
                st.markdown("  •  ".join(job['companies']))

        # ══════════════════════════════════════════════════════
        # TAB 2 — CAREER PATHWAY
        # ══════════════════════════════════════════════════════
        with tab2:

            # Fetch both data sources
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
                # ══════════════════════════════════════════════
                # SECTION 1 — NATIONAL (left) + STATE (right)
                # ══════════════════════════════════════════════
                col1, col2 = st.columns(2)

                # ── Left: National ────────────────────────────
                with col1:
                    st.markdown("**📚 Steps After Class 12**")
                    for step in pathway.get("after_class12", []):
                        st.markdown(
                            f'<div class="step-box">→ {step}</div>',
                            unsafe_allow_html=True)

                    st.markdown("")
                    st.markdown("**📝 National Entrance Exams**")
                    for exam in pathway.get("national_exams", []):
                        st.markdown(f"""
                        <div class="step-box">
                            <b>{exam.get('exam','')}</b><br>
                            <small>By {exam.get('conducted_by','')} •
                            {exam.get('frequency','')}</small>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("")
                    st.markdown("**🎓 Top Colleges (National)**")
                    for college in pathway.get("top_colleges", []):
                        st.markdown(
                            f'<div class="step-box">'
                            f'🏫 {college.get("name","")} — '
                            f'{college.get("city","")}'
                            f'</div>',
                            unsafe_allow_html=True)

                # ── Right: State ──────────────────────────────
                with col2:
                    if local_data:
                        st.markdown(f"**🏛️ Your State — {state}**")

                        # State entrance exams
                        st.markdown("")
                        st.markdown(f"**📝 State Entrance Exams**")
                        state_exams = local_data.get("state_exams", [])
                        if state_exams:
                            for exam in state_exams:
                                st.markdown(f"""
                                <div class="step-box" style="border-left-color:#e94560">
                                    <b>{exam.get('exam','')}</b><br>
                                    <small>By {exam.get('conducted_by','')} •
                                    {exam.get('eligibility','')}</small>
                                </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(
                                '<div class="step-box" style="opacity:0.6">'
                                'No state-specific exams found for this career.</div>',
                                unsafe_allow_html=True)

                        # State colleges
                        st.markdown("")
                        st.markdown(f"**🏫 Colleges in {state}**")
                        state_colleges = local_data.get("state_colleges", [])
                        if state_colleges:
                            for college in state_colleges:
                                ctype = college.get('type', '')
                                st.markdown(f"""
                                <div class="step-box" style="border-left-color:#86efac">
                                    <b>{college.get('name','')}</b><br>
                                    <small>{college.get('city','')} •
                                    <span style="color:#86efac">{ctype}</span></small>
                                </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(
                                '<div class="step-box" style="opacity:0.6">'
                                'No state colleges found for this career.</div>',
                                unsafe_allow_html=True)

                        # State scholarships
                        st.markdown("")
                        st.markdown(f"**🎓 Scholarships in {state}**")
                        scholarships = local_data.get("state_scholarships", [])
                        if scholarships:
                            for s in scholarships:
                                st.markdown(f"""
                                <div class="step-box" style="border-left-color:#fcd34d">
                                    <b>{s.get('name','')}</b><br>
                                    <small>💰 {s.get('amount','')} •
                                    {s.get('eligibility','')}</small>
                                </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(
                                '<div class="step-box" style="opacity:0.6">'
                                'Check scholarships.gov.in for state scholarships.</div>',
                                unsafe_allow_html=True)
                    else:
                        st.markdown(f"**🏛️ Your State — {state}**")
                        st.info("Could not load state-specific data. Please try again.")

                # ══════════════════════════════════════════════
                # SECTION 2 — TIMELINE (left) + PROGRESSION & SALARY (right)
                # ══════════════════════════════════════════════
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                col3, col4 = st.columns(2)

                # ── Left: Timeline ────────────────────────────
                with col3:
                    st.markdown("**📅 Year-by-Year Timeline**")
                    for t in pathway.get("timeline", []):
                        st.markdown(
                            f'<div class="step-box">'
                            f'<b>{t.get("year","")}</b> → '
                            f'{t.get("milestone","")}'
                            f'</div>',
                            unsafe_allow_html=True)

                # ── Right: Progression + Salary ───────────────
                with col4:
                    st.markdown("**📈 Career Progression**")
                    prog = pathway.get("career_progression", [])
                    for level in prog:
                        st.markdown(
                            f'<div class="step-box">→ {level}</div>',
                            unsafe_allow_html=True)

                    st.markdown("")
                    sal = pathway.get("avg_starting_salary", "")
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<h3 style="font-size:1.2rem">💰 {sal}</h3>'
                        f'<p>Average Starting Salary</p>'
                        f'</div>',
                        unsafe_allow_html=True)

                st.markdown("")
                st.caption(
                    "⚠️ AI-generated roadmap — "
                    "verify exam and college details at official websites."
                )

        st.markdown("")
        if st.button("← Back to Recommendations"):
            st.session_state.screen = 'results'
            st.rerun()


if __name__ == "__main__":
    main()
