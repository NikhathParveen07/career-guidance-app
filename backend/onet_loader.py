# ============================================
# backend/onet_loader.py
# Fetches all occupations from O*NET Web Services v2.0
#
# Verified working endpoints:
#   GET /online/occupations/                          — list all
#   GET /online/occupations/{code}/summary/interests  — RIASEC
#   GET /online/occupations/{code}/summary/skills     — skills
#
# Auth: X-API-Key header
# Stream/sector mapped from SOC code prefix (no extra API call)
# Caches in Supabase for 30 days
# ============================================
import requests
import time
from datetime import datetime, timezone
from onet_india_filter import is_india_relevant


ONET_BASE_URL = "https://api-v2.onetcenter.org"
CACHE_DAYS    = 30


# ── SOC Major Group prefix to Indian Class 12 stream ─────────
# SOC codes start with two digits indicating major occupational group
# Reference: https://www.bls.gov/soc/2018/major_groups.htm
SOC_TO_STREAM = {
    "11": "Commerce",    # Management
    "13": "Commerce",    # Business and Financial Operations
    "15": "Science",     # Computer and Mathematical
    "17": "Science",     # Architecture and Engineering
    "19": "Science",     # Life, Physical, and Social Science
    "21": "Arts",        # Community and Social Service
    "23": "Commerce",    # Legal
    "25": "Arts",        # Educational Instruction and Library
    "27": "Arts",        # Arts, Design, Entertainment, Sports, and Media
    "29": "Science",     # Healthcare Practitioners and Technical
    "31": "Science",     # Healthcare Support
    "33": "Vocational",  # Protective Service
    "35": "Vocational",  # Food Preparation and Serving Related
    "37": "Vocational",  # Building and Grounds Cleaning and Maintenance
    "39": "Vocational",  # Personal Care and Service
    "41": "Commerce",    # Sales and Related
    "43": "Commerce",    # Office and Administrative Support
    "45": "Science",     # Farming, Fishing, and Forestry
    "47": "Vocational",  # Construction and Extraction
    "49": "Vocational",  # Installation, Maintenance, and Repair
    "51": "Vocational",  # Production
    "53": "Vocational",  # Transportation and Material Moving
    "55": "Vocational",  # Military Specific
}

# ── SOC Major Group prefix to sector ─────────────────────────
SOC_TO_SECTOR = {
    "11": "Management",
    "13": "Finance",
    "15": "Technology",
    "17": "Engineering",
    "19": "Research & Science",
    "21": "Social Service",
    "23": "Legal Services",
    "25": "Education",
    "27": "Creative Arts",
    "29": "Healthcare",
    "31": "Healthcare",
    "33": "Public Safety",
    "35": "Hospitality",
    "37": "Maintenance & Repair",
    "39": "Personal Services",
    "41": "Sales",
    "43": "Administration",
    "45": "Agriculture",
    "47": "Construction",
    "49": "Maintenance & Repair",
    "51": "Manufacturing",
    "53": "Logistics & Transport",
    "55": "Military",
}


def _headers(api_key):
    """O*NET v2.0 uses X-API-Key header for authentication."""
    return {
        "Accept":    "application/json",
        "X-API-Key": api_key
    }


def _get_soc_prefix(code):
    """Extract SOC major group prefix from occupation code.
    e.g. '13-2011.00' → '13'
    """
    return code.split("-")[0] if "-" in code else "11"


def _fetch_all_occupations(api_key):
    """
    Fetch complete list of all O*NET occupations.
    Verified endpoint: GET /online/occupations/
    Returns list of {code, title} dicts.
    """
    occupations = []
    start       = 1
    page_size   = 100

    while True:
        try:
            r = requests.get(
                f"{ONET_BASE_URL}/online/occupations/",
                headers=_headers(api_key),
                params={"start": start, "end": start + page_size - 1},
                timeout=30
            )

            if r.status_code == 401:
                print("O*NET 401 — check ONET_API_KEY in Streamlit secrets")
                break

            if r.status_code != 200:
                print(f"O*NET error {r.status_code}: {r.text[:200]}")
                break

            data = r.json()
            occs = data.get("occupation", [])

            if not occs:
                break

            for occ in occs:
                occupations.append({
                    "code":  occ.get("code", ""),
                    "title": occ.get("title", "")
                })

            total = int(data.get("total", 0))
            print(f"Fetched {len(occupations)}/{total} occupations...")

            if start + page_size - 1 >= total:
                break

            start += page_size
            time.sleep(0.1)

        except Exception as e:
            print(f"Error fetching occupation list: {e}")
            break

    return occupations


def _fetch_interests(code, api_key):
    """
    Fetch RIASEC codes for one occupation.
    Verified endpoint: GET /online/occupations/{code}/summary/interests
    Uses interest_code field directly e.g. 'CEI' → C, E
    Returns (primary_riasec, secondary_riasec)
    """
    try:
        r = requests.get(
            f"{ONET_BASE_URL}/online/occupations/{code}/summary/interests",
            headers=_headers(api_key),
            timeout=15
        )
        if r.status_code != 200:
            return "R", "I"

        interest_code = r.json().get("interest_code", "")

        if len(interest_code) >= 2:
            return interest_code[0], interest_code[1]
        elif len(interest_code) == 1:
            return interest_code[0], "I"

        return "R", "I"

    except Exception:
        return "R", "I"


def _fetch_skills(code, api_key):
    """
    Fetch top 6 skills for one occupation.
    Verified endpoint: GET /online/occupations/{code}/summary/skills
    Returns list of skill name strings.
    """
    try:
        r = requests.get(
            f"{ONET_BASE_URL}/online/occupations/{code}/summary/skills",
            headers=_headers(api_key),
            params={"start": 1, "end": 6},
            timeout=15
        )
        if r.status_code != 200:
            return []

        elements = r.json().get("element", [])
        return [e.get("name", "") for e in elements if e.get("name")]

    except Exception:
        return []


def fetch_all_onet_careers(api_key, supabase):
    """
    Main function — fetches all 1016 O*NET occupations with details.

    Flow:
    1. Check Supabase cache (valid 30 days) — return if fresh
    2. Fetch all occupation codes and titles from O*NET
    3. For each occupation:
       - Fetch RIASEC from interests endpoint
       - Fetch skills from skills endpoint
       - Map stream and sector from SOC code prefix (no extra API call)
    4. Save to Supabase cache
    5. Return as list of career dicts

    Args:
        api_key  — O*NET API key (ONET_API_KEY in Streamlit secrets)
        supabase — Supabase client

    Returns list of dicts with keys:
        onet_code, job_title, sector, 12th_stream,
        primary_riasec, secondary_riasec, core_skills, cached_at
    """
    # Check Supabase cache
    try:
        cached = supabase.table("onet_careers").select("*").execute()
        if cached.data and len(cached.data) > 100:
            first      = cached.data[0]
            cached_str = first.get("cached_at", "2000-01-01T00:00:00+00:00")
            cached_str = cached_str.replace("Z", "+00:00")
            cached_at  = datetime.fromisoformat(cached_str)
            age_days   = (datetime.now(timezone.utc) - cached_at).days
            if age_days < CACHE_DAYS:
                print(f"Loaded {len(cached.data)} careers from Supabase cache")
                return cached.data
    except Exception as e:
        print(f"Cache check error: {e}")

    print("Fetching all occupations from O*NET API v2.0...")

    occupations = _fetch_all_occupations(api_key)
    print(f"Total occupations found: {len(occupations)}")
    
    # ── Apply India relevance filter ──────────────────────────
    occupations = [
        occ for occ in occupations
        if is_india_relevant(
            occ["code"],
            occ["title"],
            SOC_TO_SECTOR.get(_get_soc_prefix(occ["code"]), "General")
        )
    ]
    print(f"After India filter: {len(occupations)} occupations")


    if not occupations:
        print("No occupations returned — check ONET_API_KEY in Streamlit secrets")
        return []

    careers = []
    total   = len(occupations)

    for i, occ in enumerate(occupations):
        code   = occ["code"]
        title  = occ["title"]
        prefix = _get_soc_prefix(code)

        if (i + 1) % 100 == 0:
            print(f"Processing {i+1}/{total} occupations...")

        # Fetch RIASEC and skills from O*NET
        primary, secondary = _fetch_interests(code, api_key)
        skills             = _fetch_skills(code, api_key)

        # Map stream and sector from SOC prefix — no extra API call needed
        stream = SOC_TO_STREAM.get(prefix, "Vocational")
        sector = SOC_TO_SECTOR.get(prefix, "General")

        careers.append({
            "onet_code":        code,
            "job_title":        title,
            "sector":           sector,
            "12th_stream":      stream,
            "primary_riasec":   primary,
            "secondary_riasec": secondary,
            "core_skills":      ", ".join(skills),
            "cached_at":        datetime.now(timezone.utc).isoformat()
        })

        time.sleep(0.1)  # Respect O*NET rate limits

    print(f"Fetched details for {len(careers)} careers")

    # Save to Supabase in batches of 50
    try:
        supabase.table("onet_careers").delete().neq("onet_code", "").execute()
        for i in range(0, len(careers), 50):
            supabase.table("onet_careers").insert(careers[i:i+50]).execute()
        print(f"Saved {len(careers)} careers to Supabase")
    except Exception as e:
        print(f"Error saving to Supabase: {e}")

    return careers
