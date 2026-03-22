# ============================================
# backend/onet_loader.py
# Fetches all occupations from O*NET Web Services API v2.0
#
# Key facts from O*NET v2.0 documentation:
#   Base URL : https://api-v2.onetcenter.org
#   Auth     : X-API-Key header (NOT Basic Auth)
#   Format   : Accept: application/json
#
# Caches results in Supabase for 30 days
# ============================================
import requests
import time
from datetime import datetime, timezone


ONET_BASE_URL = "https://api-v2.onetcenter.org"
CACHE_DAYS    = 30


# ── Stream mapping from O*NET job family to Indian Class 12 stream ─
FAMILY_TO_STREAM = {
    "Architecture and Engineering":           "Science",
    "Arts and Design":                        "Arts",
    "Building and Grounds Cleaning":          "Vocational",
    "Business and Financial Operations":      "Commerce",
    "Community and Social Service":           "Arts",
    "Computer and Mathematical":              "Science",
    "Construction and Extraction":            "Vocational",
    "Education, Training, and Library":       "Arts",
    "Entertainment and Sports":               "Arts",
    "Farming, Fishing, and Forestry":         "Science",
    "Food Preparation and Serving":           "Vocational",
    "Healthcare Practitioners and Technical": "Science",
    "Healthcare Support":                     "Science",
    "Installation, Maintenance, and Repair":  "Vocational",
    "Legal":                                  "Commerce",
    "Life, Physical, and Social Science":     "Science",
    "Management":                             "Commerce",
    "Mathematics":                            "Science",
    "Media and Communication":                "Arts",
    "Military":                               "Vocational",
    "Office and Administrative Support":      "Commerce",
    "Personal Care and Service":              "Vocational",
    "Production":                             "Vocational",
    "Protective Service":                     "Vocational",
    "Sales and Related":                      "Commerce",
    "Transportation and Material Moving":     "Vocational",
}

INTEREST_TO_RIASEC = {
    "Realistic":     "R",
    "Investigative": "I",
    "Artistic":      "A",
    "Social":        "S",
    "Enterprising":  "E",
    "Conventional":  "C",
}


def _headers(api_key):
    """
    O*NET v2.0 authentication uses X-API-Key header.
    Keys are not accepted in query string or POST body.
    """
    return {
        "Accept":    "application/json",
        "X-API-Key": api_key
    }


def _fetch_all_occupations(api_key):
    """
    Fetch complete list of O*NET occupations using v2.0 API.
    Endpoint: GET /online/occupations
    Returns list of {code, title} dicts.
    """
    occupations = []
    start       = 1

    while True:
        try:
            r = requests.get(
                f"{ONET_BASE_URL}/online/occupations",
                headers=_headers(api_key),
                params={"start": start, "end": start + 99},
                timeout=30
            )

            if r.status_code == 401:
                print(f"O*NET 401 Unauthorized — check your API key in Streamlit secrets")
                break

            if r.status_code == 429:
                print("O*NET rate limit hit — waiting 500ms")
                time.sleep(0.5)
                continue

            if r.status_code != 200:
                print(f"O*NET list error {r.status_code}: {r.text[:300]}")
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

            if start + 99 >= total:
                break

            start += 100
            time.sleep(0.2)

        except Exception as e:
            print(f"Error fetching occupation list: {e}")
            break

    return occupations


def _fetch_interests(code, api_key):
    """
    Fetch RIASEC interest scores for one occupation.
    Endpoint: GET /online/occupations/{code}/interests
    Returns (primary_riasec, secondary_riasec)
    """
    try:
        r = requests.get(
            f"{ONET_BASE_URL}/online/occupations/{code}/interests",
            headers=_headers(api_key),
            timeout=15
        )

        if r.status_code == 429:
            time.sleep(0.5)
            r = requests.get(
                f"{ONET_BASE_URL}/online/occupations/{code}/interests",
                headers=_headers(api_key),
                timeout=15
            )

        if r.status_code != 200:
            return "R", "I"

        elements = r.json().get("element", [])
        scored   = []
        for elem in elements:
            name  = elem.get("name", "")
            score = float(elem.get("score", {}).get("value", 0))
            if name in INTEREST_TO_RIASEC:
                scored.append((INTEREST_TO_RIASEC[name], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        primary   = scored[0][0] if len(scored) >= 1 else "R"
        secondary = scored[1][0] if len(scored) >= 2 else "I"
        return primary, secondary

    except Exception:
        return "R", "I"


def _fetch_skills(code, api_key):
    """
    Fetch top 6 skills for one occupation.
    Endpoint: GET /online/occupations/{code}/skills
    Returns list of skill name strings.
    """
    try:
        r = requests.get(
            f"{ONET_BASE_URL}/online/occupations/{code}/skills",
            headers=_headers(api_key),
            timeout=15
        )

        if r.status_code == 429:
            time.sleep(0.5)
            r = requests.get(
                f"{ONET_BASE_URL}/online/occupations/{code}/skills",
                headers=_headers(api_key),
                timeout=15
            )

        if r.status_code != 200:
            return []

        elements = r.json().get("element", [])
        scored   = []
        for elem in elements:
            name  = elem.get("name", "")
            score = float(elem.get("score", {}).get("value", 0))
            if name:
                scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:6]]

    except Exception:
        return []


def _fetch_job_family(code, api_key):
    """
    Fetch job family for one occupation for stream mapping.
    Endpoint: GET /online/occupations/{code}
    Returns job family title string.
    """
    try:
        r = requests.get(
            f"{ONET_BASE_URL}/online/occupations/{code}",
            headers=_headers(api_key),
            timeout=15
        )

        if r.status_code == 429:
            time.sleep(0.5)
            r = requests.get(
                f"{ONET_BASE_URL}/online/occupations/{code}",
                headers=_headers(api_key),
                timeout=15
            )

        if r.status_code != 200:
            return ""

        data = r.json()

        family = data.get("job_family", {})
        if isinstance(family, dict):
            return family.get("title", "")

        cluster = data.get("career_cluster", {})
        if isinstance(cluster, dict):
            return cluster.get("title", "")

        return ""

    except Exception:
        return ""


def _map_stream(family, title):
    """Map job family to Indian Class 12 stream."""
    for key, stream in FAMILY_TO_STREAM.items():
        if key.lower() in family.lower():
            return stream

    title_lower = title.lower()
    if any(w in title_lower for w in [
        "engineer", "scientist", "doctor", "nurse", "technician",
        "biologist", "chemist", "physicist", "researcher", "medical",
        "health", "lab", "agricultural", "environmental", "software",
        "computer", "data", "cyber", "network", "programmer"
    ]):
        return "Science"
    if any(w in title_lower for w in [
        "accountant", "financial", "banker", "manager", "business",
        "sales", "marketing", "economist", "auditor", "insurance",
        "broker", "consultant", "tax", "lawyer", "attorney", "legal"
    ]):
        return "Commerce"
    if any(w in title_lower for w in [
        "teacher", "writer", "journalist", "designer", "artist",
        "social worker", "counselor", "librarian", "curator",
        "translator", "psychologist", "educator", "director",
        "presenter", "filmmaker", "animator"
    ]):
        return "Arts"

    return "Vocational"


def _map_sector(family, title):
    """Map job family to readable sector name."""
    mapping = {
        "Computer":       "Technology",
        "Mathematical":   "Technology",
        "Architecture":   "Engineering",
        "Engineering":    "Engineering",
        "Healthcare":     "Healthcare",
        "Life":           "Research & Science",
        "Physical":       "Research & Science",
        "Science":        "Research & Science",
        "Business":       "Business",
        "Financial":      "Finance",
        "Management":     "Management",
        "Legal":          "Legal Services",
        "Education":      "Education",
        "Training":       "Education",
        "Arts":           "Creative Arts",
        "Design":         "Creative Arts",
        "Media":          "Media & Publishing",
        "Communication":  "Media & Publishing",
        "Social Service": "Social Service",
        "Community":      "Social Service",
        "Farming":        "Agriculture",
        "Forestry":       "Agriculture",
        "Construction":   "Construction",
        "Extraction":     "Construction",
        "Transportation": "Logistics & Transport",
        "Food":           "Hospitality",
        "Sales":          "Sales",
        "Office":         "Administration",
        "Production":     "Manufacturing",
        "Installation":   "Maintenance & Repair",
        "Personal Care":  "Personal Services",
        "Protective":     "Public Safety",
    }

    for key, sector in mapping.items():
        if key.lower() in family.lower():
            return sector

    title_lower = title.lower()
    if any(w in title_lower for w in ["software", "data", "cyber", "IT", "network"]):
        return "Technology"
    if any(w in title_lower for w in ["doctor", "nurse", "medical", "health"]):
        return "Healthcare"
    if any(w in title_lower for w in ["teacher", "professor"]):
        return "Education"
    if any(w in title_lower for w in ["accountant", "financial", "banker"]):
        return "Finance"

    return family if family else "General"


def fetch_all_onet_careers(api_key, supabase):
    """
    Main function — fetches all O*NET occupations with full details.

    Flow:
    1. Check Supabase cache (valid 30 days) — return if fresh
    2. Fetch all occupation codes from O*NET v2.0 API
    3. For each occupation fetch interests, skills, job family
    4. Map to Indian stream and sector
    5. Save to Supabase cache
    6. Return as list of career dicts

    Args:
        api_key  — O*NET API key from Streamlit secrets (ONET_API_KEY)
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

    if not occupations:
        print("No occupations returned — check ONET_API_KEY in Streamlit secrets")
        return []

    careers = []
    total   = len(occupations)

    for i, occ in enumerate(occupations):
        code  = occ["code"]
        title = occ["title"]

        if (i + 1) % 100 == 0:
            print(f"Processing {i+1}/{total} occupations...")

        primary, secondary = _fetch_interests(code, api_key)
        skills             = _fetch_skills(code, api_key)
        family             = _fetch_job_family(code, api_key)

        stream = _map_stream(family, title)
        sector = _map_sector(family, title)

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

        time.sleep(0.1)

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
