# ============================================
# backend/pathway.py
# Generate structured Indian career roadmaps
# using Groq-hosted Llama 3.3 LLM
# ============================================
import requests
import json


GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


# ── Shared helpers ────────────────────────────────────────────

def _call_groq(prompt, groq_key):
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
                "max_tokens":  1800
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
    Prompt keys must exactly match what app.py reads:
      after_class12, courses, national_exams, national_scholarships,
      top_colleges, timeline, career_progression, avg_starting_salary,
      skills_to_develop
    """
    return f"""You are a career counsellor specialising in the Indian education system.
Generate a career roadmap for an Indian Class 12 student pursuing:
Career: {career_title}
Stream: {stream}
Sector: {sector}

Return ONLY a valid JSON object with exactly these keys, no extra text, no markdown:
{{
  "after_class12": ["step 1", "step 2", "step 3"],
  "courses": [
    {{"name": "course name", "duration": "X years", "type": "Bachelor/Diploma/Certificate"}}
  ],
  "national_exams": [
    {{"exam": "exam name", "conducted_by": "conducting body", "frequency": "once/twice a year"}}
  ],
  "national_scholarships": [
    {{"name": "scholarship name", "amount": "amount per year", "eligibility": "criteria"}}
  ],
  "top_colleges": [
    {{"name": "college name", "city": "city"}}
  ],
  "skills_to_develop": [
    {{"skill": "skill name", "level": "Beginner/Intermediate/Advanced"}}
  ],
  "timeline": [
    {{"year": "Year 1", "milestone": "what happens this year"}}
  ],
  "career_progression": ["entry level role", "mid level role", "senior level role", "leadership role"],
  "avg_starting_salary": "X-Y LPA"
}}

Rules:
- Use real Indian national exam names (JEE, NEET, CLAT, CAT, NIFT, UCEED, GATE etc.)
- Use real national colleges (IITs, NITs, AIIMS, IIMs, NLUs etc.)
- Use realistic Indian salary ranges in LPA
- List 3-5 items per array, not more
- Return ONLY the JSON object, nothing else"""


def _get_cached_pathway(career_title, supabase):
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
    try:
        supabase.table("career_pathways").upsert({
            "career_title": career_title,
            "pathway_json": json.dumps(pathway)
        }).execute()
    except Exception:
        pass


def fetch_career_pathway(career_title, stream, sector, groq_key, supabase):
    """
    Get or generate a structured national Indian career roadmap.
    Returns pathway dict or None on failure.
    """
    cached = _get_cached_pathway(career_title, supabase)
    if cached:
        # Migrate old cache entries missing new keys
        if "courses" not in cached:
            cached["courses"] = []
        if "national_scholarships" not in cached:
            cached["national_scholarships"] = []
        if "skills_to_develop" not in cached:
            cached["skills_to_develop"] = []
        return cached

    prompt   = _build_pathway_prompt(career_title, stream, sector)
    raw_text = _call_groq(prompt, groq_key)
    pathway  = _parse_json_response(raw_text)

    if pathway is None:
        return None

    # Ensure all expected keys exist even if LLM omitted some
    pathway.setdefault("after_class12", [])
    pathway.setdefault("courses", [])
    pathway.setdefault("national_exams", [])
    pathway.setdefault("national_scholarships", [])
    pathway.setdefault("top_colleges", [])
    pathway.setdefault("skills_to_develop", [])
    pathway.setdefault("timeline", [])
    pathway.setdefault("career_progression", [])
    pathway.setdefault("avg_starting_salary", "Not available")

    _save_pathway_to_cache(career_title, pathway, supabase)
    return pathway


# ── Function 2: State-Specific Recommendations ───────────────

def _build_local_prompt(career_title, stream, state):
    return f"""You are a career counsellor specialising in Indian state-level education.

For a student in {state} pursuing {career_title} from {stream} stream, provide:
1. Top 3 colleges specifically in {state} for this career
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

Rules:
- Use only real colleges and exams from {state}
- Do NOT include national exams like JEE, NEET, CLAT — only {state} specific ones
- If no state-specific exams exist, return an empty array for state_exams
- Return ONLY the JSON object"""


def fetch_local_recommendations(career_title, stream, state, groq_key, supabase):
    """
    Fetch state-specific colleges, entrance exams, and scholarships.
    Returns dict or None if state is empty or generation failed.
    """
    if not state:
        return None

    cache_key = f"{career_title}_{state}"

    try:
        cached = (supabase.table("local_recommendations")
                          .select("data_json")
                          .eq("cache_key", cache_key)
                          .execute())
        if cached.data:
            return json.loads(cached.data[0]["data_json"])
    except Exception:
        pass

    prompt   = _build_local_prompt(career_title, stream, state)
    raw_text = _call_groq(prompt, groq_key)
    data     = _parse_json_response(raw_text)

    if data is None:
        return None

    # Ensure all keys exist
    data.setdefault("state_colleges", [])
    data.setdefault("state_exams", [])
    data.setdefault("state_scholarships", [])

    try:
        supabase.table("local_recommendations").upsert({
            "cache_key": cache_key,
            "data_json": json.dumps(data)
        }).execute()
    except Exception:
        pass

    return data
