# ============================================
# app.py — Career Guidance System
# Streamlit Cloud Deployment
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import uuid
from math import log2
from datetime import datetime, timezone
from urllib.parse import quote

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title   = "CareerCompass India",
    page_icon    = "🧭",
    layout       = "wide",
    initial_sidebar_state = "collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .main { background-color: #f0f4f8; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .hero p  { font-size: 1.1rem; opacity: 0.85; margin-top: 0.5rem; }

    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 4px solid #0f3460;
    }
    .career-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #e94560;
    }
    .metric-box {
        background: #f8faff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e8f0;
    }
    .metric-box h3 { font-size: 1.6rem; color: #0f3460; margin: 0; }
    .metric-box p  { font-size: 0.8rem; color: #666; margin: 0; }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-science    { background: #dbeafe; color: #1e40af; }
    .badge-commerce   { background: #dcfce7; color: #166534; }
    .badge-arts       { background: #fce7f3; color: #9d174d; }
    .badge-vocational { background: #fef3c7; color: #92400e; }

    .riasec-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #0f3460, #e94560);
        margin: 4px 0;
    }
    .step-box {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border-left: 3px solid #0f3460;
    }
    .why-box {
        background: #fffbeb;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        border: 1px solid #fde68a;
        font-size: 0.85rem;
    }
    .demand-high { color: #dc2626; font-weight: 600; }
    .demand-mod  { color: #d97706; font-weight: 600; }
    .demand-niche{ color: #2563eb; font-weight: 600; }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #0f3460, #e94560);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }
    .stProgress > div > div { background: #0f3460; }
</style>
""", unsafe_allow_html=True)


# ── Load dependencies ─────────────────────────────────────────
@st.cache_resource
def load_models():
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone
    from supabase import create_client
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import train_test_split

    # Sentence model
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Pinecone
    pc    = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("career-discovery")

    # Supabase
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

    return sentence_model, index, supabase


@st.cache_resource
def load_svd_model():
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import train_test_split

    interactions_df = pd.read_csv("data/interactions.csv")
    reader          = Reader(rating_scale=(1, 5))
    data            = Dataset.load_from_df(
        interactions_df[['student_id', 'career_id', 'rating']], reader
    )
    trainset, _  = train_test_split(data, test_size=0.2, random_state=42)
    svd_model    = SVD(n_factors=50, random_state=42)
    svd_model.fit(trainset)
    return svd_model


@st.cache_data
def load_careers():
    return pd.read_csv("data/career_master_final.csv")


# ── RIASEC Quiz data ──────────────────────────────────────────
RIASEC_QUESTIONS = [
    {"q": "I enjoy building or fixing things with my hands",           "type": "R"},
    {"q": "I like researching facts and solving complex problems",      "type": "I"},
    {"q": "I enjoy drawing, writing, or creating original work",       "type": "A"},
    {"q": "I like teaching or helping others work through problems",   "type": "S"},
    {"q": "I enjoy leading teams or starting new initiatives",         "type": "E"},
    {"q": "I like organising data and following structured processes", "type": "C"},
    {"q": "I prefer hands-on technical work over desk work",           "type": "R"},
    {"q": "I enjoy analysing patterns and thinking logically",         "type": "I"},
    {"q": "I express myself through music, art, or creative writing",  "type": "A"},
    {"q": "I find it rewarding to support and care for others",        "type": "S"},
    {"q": "I enjoy managing, negotiating, or competing",               "type": "E"},
    {"q": "I prefer clear rules, accuracy, and working with numbers",  "type": "C"},
]

RIASEC_LABELS = {
    "R": "Realistic",   "I": "Investigative", "A": "Artistic",
    "S": "Social",      "E": "Enterprising",  "C": "Conventional"
}

RIASEC_DESC = {
    "R": "Hands-on, technical, physical work",
    "I": "Analytical, curious, research-driven",
    "A": "Creative, expressive, original",
    "S": "Caring, cooperative, people-focused",
    "E": "Ambitious, persuasive, leadership",
    "C": "Organised, detail-oriented, methodical"
}

STREAM_COLORS = {
    "Science": "badge-science", "Commerce": "badge-commerce",
    "Arts": "badge-arts",       "Vocational": "badge-vocational"
}


# ── Core recommendation functions ─────────────────────────────
def search_careers(query_text, index, sentence_model, top_k=20):
    embedding = sentence_model.encode(query_text).tolist()
    return index.query(vector=embedding, top_k=top_k,
                       include_metadata=True)


def get_riasec_boost(career_row, top2):
    pri = career_row.get('primary_riasec',   '')
    sec = career_row.get('secondary_riasec', '')
    if pri in top2 or sec in top2[0]:
        return 1.3
    elif pri == top2[1] or sec == top2[1]:
        return 1.15
    return 1.0


def get_recommendations(user_id, query, stream, riasec_top2,
                        df, index, sentence_model, svd_model,
                        is_cold_start=False, top_k=10):

    content_weight = 1.0 if is_cold_start else 0.7
    collab_weight  = 0.0 if is_cold_start else 0.3

    content_results = search_careers(query, index, sentence_model, top_k=20)

    collab_scores = {}
    if collab_weight > 0:
        for cid in range(len(df)):
            collab_scores[cid] = svd_model.predict(user_id, cid).est
        max_c = max(collab_scores.values())
        min_c = min(collab_scores.values())
        if max_c > min_c:
            for cid in collab_scores:
                collab_scores[cid] = ((collab_scores[cid] - min_c) /
                                       (max_c - min_c))
        else:
            for cid in collab_scores:
                collab_scores[cid] = 0.5

    results = []
    for match in content_results['matches']:
        try:
            cid = int(match['id'].split('_')[1])
        except (ValueError, IndexError):
            continue
        if cid >= len(df):
            continue

        row           = df.iloc[cid]
        career_stream = match['metadata']['stream']
        content_score = match['score']
        collab_score  = collab_scores.get(cid, 0.5) if collab_weight > 0 else 0.0

        if stream and career_stream != stream:
            collab_score *= 0.5

        stream_boost = 1.2 if career_stream == stream else 1.0
        riasec_boost = get_riasec_boost(row.to_dict(), riasec_top2)
        final_score  = (content_weight * content_score +
                        collab_weight  * collab_score) * stream_boost * riasec_boost

        results.append({
            'career_id':        cid,
            'career':           row['job_title'],
            'stream':           career_stream,
            'sector':           row['sector'],
            'primary_riasec':   row['primary_riasec'],
            'secondary_riasec': row['secondary_riasec'],
            'core_skills':      row['core_skills'],
            'final_score':      round(final_score,   4),
            'content_score':    round(content_score, 4),
            'collab_score':     round(collab_score,  4),
            'stream_boost':     stream_boost,
            'riasec_boost':     riasec_boost
        })

    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results[:top_k]


def generate_explanation(rec, query, riasec_top2, student_stream):
    query_words   = set(query.lower().split()) - {
        'and','or','the','a','an','in','of','to','with','for','is'
    }
    skill_words   = set(rec['core_skills'].lower().replace(',','').split())
    overlap       = query_words & skill_words

    if overlap:
        interest = f"Your interest in '{', '.join(sorted(overlap))}' matches this career's core skills."
    elif rec['content_score'] >= 0.5:
        interest = f"Strong semantic match ({rec['content_score']:.0%}) with your interest profile."
    else:
        interest = f"Broad alignment ({rec['content_score']:.0%}) — this career expands your interest area."

    boost = rec['riasec_boost']
    pri   = rec['primary_riasec']
    if boost == 1.3:
        riasec = f"Strong personality fit — your {RIASEC_LABELS[riasec_top2[0]]} trait aligns with this career."
    elif boost == 1.15:
        riasec = f"Moderate personality fit — your {RIASEC_LABELS[riasec_top2[1]]} trait fits this career."
    else:
        riasec = f"This career may stretch your comfort zone — suits {RIASEC_LABELS.get(pri,pri)} profiles."

    stream = (f"Directly aligned with your {student_stream} stream." 
              if rec['stream_boost'] > 1.0 
              else f"From {rec['stream']} stream — accessible with bridging courses.")

    return interest, riasec, stream


def fetch_job_data(career_title, serpapi_key):
    cached = st.session_state.get(f"job_{career_title}")
    if cached:
        return cached

    cities    = ["Bangalore", "Mumbai", "Delhi"]
    all_jobs  = []
    companies = []
    salaries  = []

    for city in cities:
        try:
            r = requests.get(
                "https://serpapi.com/search",
                params={
                    "engine":  "google_jobs",
                    "q":       f"{career_title} jobs in {city} India",
                    "hl":      "en", "gl": "in",
                    "api_key": serpapi_key
                }, timeout=10
            )
            data = r.json()
            if "error" in data:
                break
            jobs = data.get("jobs_results", [])
            all_jobs.extend(jobs)
            for job in jobs:
                if job.get("company_name"):
                    companies.append(job["company_name"])
                sal = job.get("detected_extensions", {}).get("salary")
                if sal:
                    salaries.append(sal)
            time.sleep(0.5)
        except Exception:
            continue

    total  = len(all_jobs)
    demand = ("🔥 High Demand"    if total >= 15 else
              "📈 Moderate Demand" if total >= 7  else
              "📊 Niche Market"    if total > 0   else
              "🔍 Limited Data")

    result = {
        "total":     total,
        "demand":    demand,
        "companies": list(dict.fromkeys(companies))[:5],
        "salary":    salaries[0] if salaries else "Not available"
    }
    st.session_state[f"job_{career_title}"] = result
    return result


def fetch_pathway(career_title, stream, sector, groq_key, supabase):
    try:
        cached = (supabase.table("career_pathways")
                          .select("pathway_json")
                          .eq("career_title", career_title)
                          .execute())
        if cached.data:
            return json.loads(cached.data[0]["pathway_json"])
    except Exception:
        pass

    prompt = f"""You are a career counsellor for Indian Class 12 students.
Generate a career roadmap for: {career_title} | Stream: {stream} | Sector: {sector}
Return ONLY valid JSON with these exact keys:
{{"after_class12":["step1","step2"],"national_exams":[{{"exam":"","conducted_by":"","frequency":""}}],"state_exams":[{{"exam":"","state_or_university":""}}],"top_colleges":[{{"name":"","city":""}}],"timeline":[{{"year":"","milestone":""}}],"first_job_titles":[""],"career_progression":[""],"avg_starting_salary":""}}
Use real Indian exams, colleges, salary. Return ONLY JSON."""

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}",
                     "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3, "max_tokens": 1500},
            timeout=30
        )
        raw   = r.json()["choices"][0]["message"]["content"]
        clean = raw.replace("```json","").replace("```","").strip()
        try:
            start   = clean.index("{")
            end     = clean.rindex("}") + 1
            pathway = json.loads(clean[start:end])
        except Exception:
            pathway = json.loads(clean)

        supabase.table("career_pathways").upsert({
            "career_title": career_title,
            "pathway_json": json.dumps(pathway)
        }).execute()
        return pathway
    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════
def main():
    # Load resources
    sentence_model, index, supabase = load_models()
    svd_model                       = load_svd_model()
    df                              = load_careers()

    # Session state init
    if 'screen'       not in st.session_state:
        st.session_state.screen = 'profile'
    if 'profile'      not in st.session_state:
        st.session_state.profile = {}
    if 'riasec'       not in st.session_state:
        st.session_state.riasec = {}
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'results'      not in st.session_state:
        st.session_state.results = []
    if 'selected_career' not in st.session_state:
        st.session_state.selected_career = None

    # ── HERO ──────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🧭 CareerCompass India</h1>
        <p>Discover careers that match your stream, personality, and interests</p>
    </div>
    """, unsafe_allow_html=True)

    # ── PROGRESS BAR ──────────────────────────────────────────
    screens = ['profile', 'quiz', 'results', 'detail']
    steps   = ['📋 Profile', '🧭 RIASEC Quiz',
               '🎯 Recommendations', '🗺️ Career Detail']
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
                name   = st.text_input("Your Name", placeholder="e.g. Riya Sharma")
                stream = st.selectbox("Class 12 Stream",
                    ["Science", "Commerce", "Arts", "Vocational"])
                marks  = st.slider("Your Marks (%)", 40, 100, 75)

            with col2:
                city   = st.text_input("Your City", placeholder="e.g. Hyderabad")
                budget = st.selectbox("Annual Education Budget",
                    ["Under ₹50,000", "₹50,000–₹1.5L",
                     "₹1.5L–₹5L",    "Above ₹5L"])
                query  = st.text_area("What are your interests?",
                    placeholder="e.g. I love biology, drawing, and helping people",
                    height=80)

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
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Q{i+1}.** {q['q']}")
                with col2:
                    answers[i] = st.select_slider(
                        f"q{i}", options=[1,2,3,4,5],
                        value=3, label_visibility="collapsed"
                    )

            submitted = st.form_submit_button("Get My Recommendations →")

            if submitted:
                scores = {"R":0,"I":0,"A":0,"S":0,"E":0,"C":0}
                for i, q in enumerate(RIASEC_QUESTIONS):
                    scores[q['type']] += answers[i]

                ranked = sorted(scores.items(),
                                key=lambda x: x[1], reverse=True)
                top2   = [ranked[0][0], ranked[1][0]]

                st.session_state.riasec = {
                    "scores":      scores,
                    "ranked":      ranked,
                    "top2":        top2,
                    "riasec_code": "".join(top2)
                }
                st.session_state.screen = 'results'
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

        # ── RIASEC Profile Summary ────────────────────────────
        st.markdown("### 🎯 Your Career Recommendations")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-box">
                <h3>{profile['stream']}</h3><p>Your Stream</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-box">
                <h3>{riasec['riasec_code']}</h3><p>RIASEC Code</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-box">
                <h3>{profile['marks']}%</h3><p>Your Marks</p>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-box">
                <h3>{RIASEC_LABELS[riasec['top2'][0]]}</h3>
                <p>Primary Personality</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── RIASEC Bar Chart ──────────────────────────────────
        with st.expander("📊 View Your Full RIASEC Profile"):
            for rtype, score in riasec['ranked']:
                pct = int((score / 10) * 100)
                st.markdown(
                    f"**{RIASEC_LABELS[rtype]}** ({rtype}) — {score}/10"
                )
                st.progress(pct / 100)
                st.caption(RIASEC_DESC[rtype])

        st.markdown("---")

        # ── Generate recommendations ──────────────────────────
        if not st.session_state.results:
            with st.spinner("🔄 Finding your best career matches..."):
                recs = get_recommendations(
                    user_id        = profile['student_id'],
                    query          = profile['query'],
                    stream         = profile['stream'],
                    riasec_top2    = riasec['top2'],
                    df             = df,
                    index          = index,
                    sentence_model = sentence_model,
                    svd_model      = svd_model,
                    is_cold_start  = True,
                    top_k          = 10
                )
                st.session_state.results = recs

        results = st.session_state.results

        # ── Career Cards ──────────────────────────────────────
        st.markdown(f"#### 🏆 Top {len(results)} Careers for You")

        for i, rec in enumerate(results):
            stream_badge = STREAM_COLORS.get(rec['stream'], 'badge-science')
            riasec_icon  = ("⭐⭐" if rec['riasec_boost'] == 1.3  else
                            "⭐"   if rec['riasec_boost'] == 1.15 else "")
            stream_icon  = "🎓" if rec['stream_boost'] > 1.0 else ""

            interest, riasec_exp, stream_exp = generate_explanation(
                rec, profile['query'], riasec['top2'], profile['stream']
            )

            with st.container():
                st.markdown(f"""
                <div class="career-card">
                    <div style="display:flex; justify-content:space-between; align-items:center">
                        <div>
                            <span style="font-size:1.1rem; font-weight:700">
                                #{i+1} {rec['career']}
                            </span>
                            {stream_icon}{riasec_icon}
                        </div>
                        <div style="text-align:right">
                            <span style="font-size:1.3rem; font-weight:700;
                                         color:#0f3460">
                                {int(rec['final_score']*100)}%
                            </span>
                            <br>
                            <small style="color:#888">match score</small>
                        </div>
                    </div>
                    <div style="margin-top:0.5rem">
                        <span class="badge {stream_badge}">{rec['stream']}</span>
                        <span class="badge" style="background:#f0f4f8;color:#555">
                            {rec['sector']}
                        </span>
                        <span class="badge" style="background:#e8f4fd;color:#0f3460">
                            RIASEC: {rec['primary_riasec']}/{rec['secondary_riasec']}
                        </span>
                    </div>
                    <div class="why-box">
                        💡 <b>Why this fits you:</b><br>
                        • {interest}<br>
                        • {riasec_exp}<br>
                        • {stream_exp}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button(f"View Details →",
                                 key=f"detail_{i}"):
                        st.session_state.selected_career = rec
                        st.session_state.screen = 'detail'
                        st.rerun()

        if st.button("← Start Over"):
            for key in ['screen','profile','riasec',
                        'results','selected_career','quiz_answers']:
                if key in st.session_state:
                    del st.session_state[key]
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

        st.markdown(f"### 🗺️ {rec['career']}")
        st.markdown(f"""
        <span class="badge {STREAM_COLORS.get(rec['stream'],'badge-science')}">
            {rec['stream']}
        </span>
        <span class="badge" style="background:#f0f4f8;color:#555">
            {rec['sector']}
        </span>
        <span class="badge" style="background:#e8f4fd;color:#0f3460">
            RIASEC: {rec['primary_riasec']}/{rec['secondary_riasec']}
        </span>
        """, unsafe_allow_html=True)

        st.markdown("")

        tab1, tab2 = st.tabs(["📊 Job Market", "🗺️ Career Pathway"])

        # ── Tab 1: Job Market ─────────────────────────────────
        with tab1:
            with st.spinner("🌐 Fetching live job data..."):
                job = fetch_job_data(
                    rec['career'],
                    st.secrets["SERPAPI_KEY"]
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class="metric-box">
                    <h3>{job['total']}+</h3>
                    <p>Sample Openings (3 cities)</p>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="metric-box">
                    <h3 style="font-size:1rem">{job['demand']}</h3>
                    <p>Market Demand</p>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="metric-box">
                    <h3 style="font-size:1rem">{job['salary']}</h3>
                    <p>Salary Range</p>
                </div>""", unsafe_allow_html=True)

            if job['companies']:
                st.markdown("**🏢 Top Hiring Companies:**")
                st.markdown(" • ".join(job['companies']))

        # ── Tab 2: Career Pathway ─────────────────────────────
        with tab2:
            with st.spinner("🤖 Generating career roadmap..."):
                pathway = fetch_pathway(
                    rec['career'], rec['stream'],
                    rec['sector'],
                    st.secrets["GROQ_API_KEY"],
                    supabase
                )

            if pathway:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📚 Steps After Class 12**")
                    for step in pathway.get("after_class12", []):
                        st.markdown(
                            f'<div class="step-box">→ {step}</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("")
                    st.markdown("**📝 National Entrance Exams**")
                    for exam in pathway.get("national_exams", []):
                        st.markdown(f"""
                        <div class="step-box">
                            <b>{exam.get('exam','')}</b><br>
                            <small>By {exam.get('conducted_by','')} •
                            {exam.get('frequency','')}</small>
                        </div>""", unsafe_allow_html=True)

                    if pathway.get("state_exams"):
                        st.markdown("")
                        st.markdown("**🏛️ State Admissions**")
                        for exam in pathway.get("state_exams", []):
                            st.markdown(
                                f'<div class="step-box">'
                                f'{exam.get("exam","")} — '
                                f'{exam.get("state_or_university","")}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                with col2:
                    st.markdown("**🎓 Top Colleges**")
                    for college in pathway.get("top_colleges", []):
                        st.markdown(
                            f'<div class="step-box">'
                            f'🏫 {college.get("name","")} — '
                            f'{college.get("city","")}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("")
                    st.markdown("**📅 Timeline**")
                    for t in pathway.get("timeline", []):
                        st.markdown(
                            f'<div class="step-box">'
                            f'<b>{t.get("year","")}</b> → '
                            f'{t.get("milestone","")}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("")
                    st.markdown("**📈 Career Progression**")
                    prog = pathway.get("career_progression", [])
                    st.markdown(
                        f'<div class="step-box">'
                        f'{" → ".join(prog)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown("")
                    sal = pathway.get("avg_starting_salary","")
                    st.markdown(f"""<div class="metric-box">
                        <h3 style="font-size:1.1rem">💰 {sal}</h3>
                        <p>Average Starting Salary</p>
                    </div>""", unsafe_allow_html=True)

                st.caption(
                    "⚠️ AI-generated roadmap — "
                    "verify exam details at official websites."
                )
            else:
                st.warning("Could not generate pathway. Please try again.")

        st.markdown("")
        if st.button("← Back to Recommendations"):
            st.session_state.screen = 'results'
            st.rerun()


if __name__ == "__main__":
    main()
