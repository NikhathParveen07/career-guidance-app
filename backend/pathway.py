# ============================================
# backend/pathway.py
# Generate structured Indian career roadmaps
# using Groq-hosted Llama 3.3 LLM
# Two functions:
#   1. fetch_career_pathway        — national roadmap
#   2. fetch_local_recommendations — state-specific colleges + exams
# Results cached in Supabase for reuse across sessions
# ============================================
import requests
import json


GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


# ── Shared helpers ────────────────────────────────────────────

def _call_groq(prompt, groq_key):
    """
    Call the Groq API with a prompt and return raw text response.
    Shows Streamlit error messages on failure for easy debugging.
    Returns None on any failure.
    """
    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type":  "application/json"
            },
            json={
                "model":       GROQ_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens":  1500
            },
            timeout=30
        )

        if response.status_code != 200:
            import streamlit as st
            st.error(f"Groq API error {response.status_code}: {response.text[:300]}")
            return None

        data = response.json()

        if "error" in data:
            import streamlit as st
            st.error(f"Groq error: {data['error']}")
            return None

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        import streamlit as st
        st.error(f"Groq exception: {str(e)}")
        return None


def _parse_json_response(raw_text):
    """
    Parse JSON from LLM response.
    Handles markdown code fences and extracts first JSON object
    as fallback when the model adds extra text around the JSON.
    """
    if not raw_text:
        return None

    clean = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    try:
        start = clean.index("{")
        end   = clean.rindex("}") + 1
        return json.loads(clean[start:end])
    except Exception:
        return None


# ── Function 1: National Career Pathway ──────────────────────

def _build_pathway_prompt(career_title, stream, sector):
    """
    Build prompt for national Indian career roadmap.
    Requests real exam names, real colleges, real salary ranges.
    """
    return f"""You are a career counsellor specialising in the Indian education system.
Generate a detailed career roadmap for an Indian Class 12 student pursuing:
Career: {career_title}
Stream: {stream}
Sector: {sector}

Return ONLY a valid JSON object with exactly these keys, no extra text, no markdown:
{{
  "after_class12": ["step 1", "step 2", "step 3"],
  "national_exams": [
    {{"exam": "exam name", "conducted_by": "conducting body", "frequency": "once/twice a year"}}
  ],
  "state_exams": [
    {{"exam": "exam name", "state_or_university": "name"}}
  ],
  "top_colleges": [
    {{"name": "college name", "city": "city"}}
  ],
  "timeline": [
    {{"year": "Year 1", "milestone": "what happens this year"}}
  ],
  "first_job_titles": ["title 1", "title 2", "title 3"],
  "career_progression": ["entry level", "mid level", "senior level", "leadership"],
  "avg_starting_salary": "X-Y LPA"
}}

Use real Indian exam names (JEE, NEET, CLAT, CAT, NIFT, UCEED etc.),
real colleges (IITs, NITs, AIIMS, IIMs etc.), and realistic Indian salary ranges.
Return ONLY the JSON object."""


def _get_cached_pathway(career_title, supabase):
    """Check Supabase for a previously generated national pathway."""
    try:
        result = (supabase
                  .table("career_pathways")
                  .select("pathway_json")
                  .eq("career_title", career_title)
                  .execute())
        if result.data:
            return json.loads(result.data[0]["pathway_json"])
    except Exception:
        pass
    return None


def _save_pathway_to_cache(career_title, pathway, supabase):
    """Save a generated national pathway to Supabase cache."""
    try:
        supabase.table("career_pathways").upsert({
            "career_title": career_title,
            "pathway_json": json.dumps(pathway)
        }).execute()
    except Exception:
        pass


def fetch_career_pathway(career_title, stream, sector, groq_key, supabase):
    """
    Get or generate a structured Indian career roadmap.

    Strategy:
    1. Check Supabase cache - return immediately if found
    2. Generate fresh pathway from Groq LLM
    3. Cache result in Supabase for all future students
    4. Return pathway dict or None on failure
    """
    cached = _get_cached_pathway(career_title, supabase)
    if cached:
        return cached

    prompt   = _build_pathway_prompt(career_title, stream, sector)
    raw_text = _call_groq(prompt, groq_key)
    pathway  = _parse_json_response(raw_text)

    if pathway is None:
        return None

    _save_pathway_to_cache(career_title, pathway, supabase)
    return pathway


# ── Function 2: State-Specific Recommendations ───────────────

def _build_local_prompt(career_title, stream, state):
    """
    Build prompt for state-specific colleges, exams, and scholarships.
    """
    return f"""You are a career counsellor specialising in Indian state-level education.

For a student in {state} pursuing {career_title} from {stream} stream, provide:
1. Top 3 colleges in {state} for this career
2. State-level entrance exams conducted in {state} for this career
3. State government scholarships available in {state}

Return ONLY valid JSON with exactly these keys, no extra text:
{{
  "state_colleges": [
    {{"name": "college name", "city": "city in {state}", "type": "Government/Private/Deemed"}}
  ],
  "state_exams": [
    {{"exam": "exam name", "conducted_by": "state board or university", "eligibility": "who can apply"}}
  ],
  "state_scholarships": [
    {{"name": "scholarship name", "amount": "amount per year", "eligibility": "criteria"}}
  ]
}}

Use only real colleges and exams from {state}. Return ONLY the JSON."""


def fetch_local_recommendations(career_title, stream, state, groq_key, supabase):
    """
    Fetch state-specific colleges, entrance exams, and scholarships
    based on the student's selected state from the profile form.

    State is passed directly - no city-to-state mapping needed.
    Cached in Supabase by career + state combination.

    Args:
        career_title — career to generate local recommendations for
        stream       — student's Class 12 stream
        state        — student's state (from profile form dropdown)
        groq_key     — Groq API key from Streamlit secrets
        supabase     — Supabase client from data_loader

    Returns:
        dict with state_colleges, state_exams, state_scholarships
        or None if state is empty or generation failed
    """
    if not state:
        return None

    cache_key = f"{career_title}_{state}"

    # Check Supabase cache
    try:
        cached = (supabase.table("local_recommendations")
                          .select("data_json")
                          .eq("cache_key", cache_key)
                          .execute())
        if cached.data:
            return json.loads(cached.data[0]["data_json"])
    except Exception:
        pass

    # Generate from Groq
    prompt   = _build_local_prompt(career_title, stream, state)
    raw_text = _call_groq(prompt, groq_key)
    data     = _parse_json_response(raw_text)

    if data is None:
        return None

    # Cache result
    try:
        supabase.table("local_recommendations").upsert({
            "cache_key": cache_key,
            "data_json": json.dumps(data)
        }).execute()
    except Exception:
        pass

    return data
