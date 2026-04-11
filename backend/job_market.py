# ============================================
# backend/job_market.py
# ============================================
import requests
import time
import streamlit as st

from backend.future_market import get_future_market_data  # proper package import

INDIAN_CITIES    = ["Bangalore", "Mumbai", "Delhi"]
SERPAPI_ENDPOINT = "https://serpapi.com/search"


def fetch_full_market_data(career_title, sector, stream,
                           serpapi_key, groq_key,
                           news_api_key, supabase):

    cache_key = f"job_{career_title}"
    if cache_key in st.session_state:
        current = st.session_state[cache_key]
    else:
        all_jobs  = []
        companies = []
        salaries  = []
        locations = []

        # Only attempt SerpAPI if a key is provided
        if serpapi_key:
            for city in INDIAN_CITIES:
                try:
                    r = requests.get(
                        SERPAPI_ENDPOINT,
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
                        loc = job.get("location")
                        if loc:
                            city_name = loc.split(",")[0].strip()
                            if city_name:
                                locations.append(city_name)
                    time.sleep(0.5)
                except Exception:
                    continue

        current = {
            "total":         len(all_jobs),
            "salary_string": salaries[0] if salaries else "Not available",
            "companies":     list(dict.fromkeys(companies))[:5],
            "locations":     list(dict.fromkeys(locations))[:5]
        }
        st.session_state[cache_key] = current

    future = get_future_market_data(
        career_title          = career_title,
        sector                = sector,
        stream                = stream,
        serpapi_salary_string = current["salary_string"],
        serpapi_companies     = current["companies"],
        serpapi_locations     = current["locations"],
        groq_key              = groq_key,
        news_api_key          = news_api_key,
        supabase              = supabase
    )

    return {"current": current, "future": future}
