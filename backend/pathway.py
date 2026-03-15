# ============================================
# backend/pathway.py
# Generate structured Indian career roadmaps
# using Groq-hosted Llama 3.3 LLM
# Results cached in Supabase for reuse across sessions
# ============================================
import requests
import json


GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def _build_prompt(career_title, stream, sector):
    """
    Build a structured prompt instructing the LLM to return
    a JSON career roadmap specific to the Indian educational system.

    The prompt explicitly requests real Indian exam names,
    real colleges, and real salary benchmarks to reduce hallucination.
    Structured JSON output is enforced to allow reliable parsing.
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
  "avg_starting_salary": "X–Y LPA"
}}

Use real Indian exam names (JEE, NEET, CLAT, CAT, NIFT, UCEED etc.),
real colleges (IITs, NITs, AIIMS, IIMs etc.), and realistic Indian salary ranges.
Return ONLY the JSON object."""


def _call_groq(prompt, groq_key):
    """
    Call the Groq API with a prompt and return the raw text response.
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

        # Show status for debugging
        if response.status_code != 200:
            import streamlit as st
            st.error(f"Groq API error {response.status_code}: {response.text[:300]}")
            return None

        data = response.json()

        # Check for API-level error
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
    Handles cases where the model wraps JSON in markdown code fences.
    Falls back to extracting the first JSON object if direct parse fails.
    """
    if not raw_text:
        return None

    # Strip markdown fences if present
    clean = raw_text.replace("```json", "").replace("```", "").strip()

    # Attempt direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON object by finding first { and last }
    try:
        start = clean.index("{")
        end   = clean.rindex("}") + 1
        return json.loads(clean[start:end])
    except Exception:
        return None


def _get_cached_pathway(career_title, supabase):
    """
    Check Supabase for a previously generated pathway.
    Returns parsed pathway dict or None if not cached.
    """
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
    """
    Save a generated pathway to Supabase cache.
    Uses upsert so re-generation overwrites stale data.
    """
    try:
        supabase.table("career_pathways").upsert({
            "career_title": career_title,
            "pathway_json": json.dumps(pathway)
        }).execute()
    except Exception:
        pass  # Cache failure is non-fatal


def fetch_career_pathway(career_title, stream, sector, groq_key, supabase):
    """
    Main function: get or generate a structured Indian career roadmap.

    Strategy:
    1. Check Supabase cache — return immediately if found
       (saves API calls; same career benefits all future students)
    2. Generate fresh pathway from Groq LLM
    3. Cache the result in Supabase for future use
    4. Return pathway dict or None on failure

    The disclaimer about AI-generated content is handled in the UI layer.

    Args:
        career_title — career to generate pathway for
        stream       — student's Class 12 stream
        sector       — career's industrial sector
        groq_key     — Groq API key from Streamlit secrets
        supabase     — Supabase client from data_loader

    Returns:
        dict with pathway data, or None if generation failed
    """
    # Step 1 — Check cache
    cached = _get_cached_pathway(career_title, supabase)
    if cached:
        return cached

    # Step 2 — Generate from LLM
    prompt   = _build_prompt(career_title, stream, sector)
    raw_text = _call_groq(prompt, groq_key)
    pathway  = _parse_json_response(raw_text)

    if pathway is None:
        return None

    # Step 3 — Cache for future use
    _save_pathway_to_cache(career_title, pathway, supabase)

    return pathway
