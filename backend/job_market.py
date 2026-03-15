# ============================================
# backend/job_market.py
# Fetch live Indian job market data via SerpAPI
# Results cached in session state to minimise API calls
# ============================================
import requests
import time
import streamlit as st


INDIAN_CITIES    = ["Bangalore", "Mumbai", "Delhi"]
SERPAPI_ENDPOINT = "https://serpapi.com/search"


def _get_demand_level(total_openings):
    """
    Classify market demand based on number of sample job listings.

    Note: SerpAPI returns max 10 results per city query.
    So a total of 30 = 10 per city across 3 cities.
    Actual national market is larger — this is a sample indicator.
    """
    if total_openings >= 15:
        return "🔥 High Demand"
    elif total_openings >= 7:
        return "📈 Moderate Demand"
    elif total_openings > 0:
        return "📊 Niche Market"
    else:
        return "🔍 Limited Data"


def fetch_job_market_data(career_title, serpapi_key):
    """
    Fetch live job market data for a career from Indian cities.

    Strategy:
    1. Check session state cache — return immediately if found
    2. Query SerpAPI Google Jobs across 3 Indian cities
    3. Extract job count, company names, salary data
    4. Cache result in session state for the session duration
    5. On API limit or error, return a graceful empty result

    Args:
        career_title — job title to search for
        serpapi_key  — SerpAPI API key from Streamlit secrets

    Returns:
        dict with keys: total, demand, companies, salary
    """
    # Check session cache first — avoids re-querying same career
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

            # Detect API limit or error
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

            time.sleep(0.5)  # Respect SerpAPI rate limits

        except Exception:
            continue  # Skip failed city, try next

    result = {
        "total":     len(all_jobs),
        "demand":    _get_demand_level(len(all_jobs)),
        "companies": list(dict.fromkeys(companies))[:5],  # unique, preserve order
        "salary":    salaries[0] if salaries else "Not available"
    }

    # Save to session cache
    st.session_state[cache_key] = result
    return result
