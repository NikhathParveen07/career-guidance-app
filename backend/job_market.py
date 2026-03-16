# ============================================
# backend/job_market.py
# Current job market data via SerpAPI
# + Future market data via future_market.py
# ============================================
import requests
import time
import streamlit as st
from backend.future_market import get_future_market_data

INDIAN_CITIES    = ["Bangalore", "Mumbai", "Delhi"]
SERPAPI_ENDPOINT = "https://serpapi.com/search"


def _get_demand_level(total_openings):
    """Classify current market activity from job count sample."""
    if total_openings >= 15:
        return "🔥 High"
    elif total_openings >= 7:
        return "📈 Moderate"
    elif total_openings > 0:
        return "📊 Niche"
    else:
        return "🔍 Limited Data"


def fetch_current_job_data(career_title, serpapi_key):
    """
    Fetch current job listings snapshot from SerpAPI Google Jobs.
    Cached in session state for the session duration.

    Returns current market snapshot only.
    Salary string is passed to future_market.py for projection.
    """
    cache_key = f"job_{career_title}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    all_jobs  = []
    companies = []
    salaries  = []

    for city in INDIAN_CITIES:
        try:
            response = requests.get(
                SERPAPI_ENDPOINT,
                params={
                    "engine":  "google_jobs",
                    "q":       f"{career_title} jobs in {city} India",
                    "hl":      "en",
                    "gl":      "in",
                    "api_key": serpapi_key
                },
                timeout=10
            )
            data = response.json()
            if "error" in data:
                break
            jobs = data.get("jobs_results", [])
            all_jobs.extend(jobs)
            for job in jobs:
                if job.get("company_name"):
                    companies.append(job["company_name"])
                salary = job.get("detected_extensions", {}).get("salary")
                if salary:
                    salaries.append(salary)
            time.sleep(0.5)
        except Exception:
            continue

    result = {
        "total":          len(all_jobs),
        "demand":         _get_demand_level(len(all_jobs)),
        "companies":      list(dict.fromkeys(companies))[:5],
        "salary_string":  salaries[0] if salaries else "Not available"
    }
    st.session_state[cache_key] = result
    return result


def fetch_full_market_data(career_title, sector, stream,
                           serpapi_key, groq_key,
                           news_api_key, supabase):
    """
    Combined market intelligence function.

    Step 1 — Fetch current snapshot via SerpAPI:
        Total job listings, hiring companies, current salary string

    Step 2 — Fetch future outlook via future_market.py:
        Salary projection at graduation (World Bank growth rate)
        Demand signal (Google News RSS + NewsAPI + Groq LLM)

    The SerpAPI salary string from Step 1 is passed directly
    into Step 2 as the current salary baseline — no extra
    API or card required.

    Args:
        career_title  — career name
        sector        — career sector
        stream        — student stream
        serpapi_key   — SerpAPI key
        groq_key      — Groq API key
        news_api_key  — NewsAPI key (can be None — RSS still works)
        supabase      — Supabase client

    Returns:
        dict with "current" and "future" keys
    """
    # Step 1 — Current snapshot
    current = fetch_current_job_data(career_title, serpapi_key)

    # Step 2 — Future outlook
    # Pass SerpAPI salary string as current salary baseline
    future = get_future_market_data(
        career_title          = career_title,
        sector                = sector,
        stream                = stream,
        serpapi_salary_string = current["salary_string"],
        groq_key              = groq_key,
        news_api_key          = news_api_key,
        supabase              = supabase
    )

    return {
        "current": current,
        "future":  future
    }
