# ============================================
# backend/onet_loader.py
# Fetches all occupations from O*NET Web Services API
# Correct endpoints from O*NET API v2.0 documentation
# Maps RIASEC, skills, sector, and Indian stream
# Caches in Supabase for 30 days
# ============================================
import requests
import json
import time
from datetime import datetime, timezone
from requests.auth import HTTPBasicAuth


ONET_BASE_URL = "https://services.onetcenter.org/ws"
CACHE_DAYS    = 30


# ── Stream mapping from O*NET job family to Indian Class 12 stream ─
FAMILY_TO_STREAM = {
    # Science stream
    "Architecture and Engineering":              "Science",
    "Arts and Design":                           "Arts",
    "Building and Grounds Cleaning":             "Vocational",
    "Business and Financial Operations":         "Commerce",
    "Community and Social Service":              "Arts",
    "Computer and Mathematical":                 "Science",
    "Construction and Extraction":               "Vocational",
    "Education, Training, and Library":          "Arts",
    "Entertainment and Sports":                  "Arts",
    "Farming, Fishing, and Forestry":            "Science",
    "Food Preparation and Serving":              "Vocational",
    "Healthcare Practitioners and Technical":    "Science",
    "Healthcare Support":                        "Science",
    "Installation, Maintenance, and Repair":     "Vocational",
    "Legal":                                     "Commerce",
    "Life, Physical, and Social Science":        "Science",
    "Management":                                "Commerce",
    "Mathematics":                               "Science",
    "Media and Communication":                   "Arts",
    "Military":                                  "Vocational",
    "Office and Administrative Support":         "Commerce",
    "Personal Care and Service":                 "Vocational",
    "Production":                                "Vocational",
    "Protective Service":                        "Vocational",
    "Sales and Related":                         "Commerce",
    "Transportation and Material Moving":        "Vocational",
}

# ── RIASEC letter mapping ─────────────────────────────────────────
INTEREST_TO_RIASEC = {
    "Realistic":     "R",
    "Investigative": "I",
    "Artistic":      "A",
    "Social":        "S",
    "Enterprising":  "E",
    "Conventional":  "C",
}


def _auth(username, password):
    return HTTPBasicAuth(username, password)


def _headers():
    return {"Accept": "application/json"}


def _fetch_all_occupations(username, password):
    """
    Fetch complete list of O*NET occupations using the
    correct O*NET OnLine browse endpoint.

    Endpoint: GET /online/occupations
    Returns list of {code, title} dicts.
    """
    occupations = []
    start = 1

    while True:
        try:
            r = requests.get(
                f"{ONET_BASE_URL}/online/occupations",
                auth=_auth(username, password),
                headers=_headers(),
                params={"start": start, "end": start + 99},
                timeout=30
            )

            if r.status_code != 200:
                print(f"O*NET list error {r.status_code}: {r.text[:200]}")
                break

            data  = r.json()
            occs  = data.get("occupation", [])

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


def _fetch_interests(code, username, password):
    """
    Fetch RIASEC interest scores for one occupation.
    Endpoint: GET /online/occupations/{code}/interests
    Returns (primary_riasec, secondary_riasec)
    """
    try:
        r = requests.get(
            f"{ONET_BASE_URL}/online/occupations/{code}/interests",
            auth=_auth(username, password),
            headers=_headers(),
            timeout=15
        )

        if r.status_code != 200:
            return "R", "I"

        data     = r.json()
        elements = data.get("element", [])

        scored = []
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


def _fetch_skills(code, username, password):
    """
    Fetch top 6 skills for one occupation.
    Endpoint: GET /online/occupations/{code}/skills
    Returns list of skill name strings.
    """
    try:
        r = requests.get(
            f"{ONET_BASE_URL}/online/occupations/{code}/skills",
            auth=_auth(username, password),
            headers=_headers(),
            timeout=15
        )

        if r.status_code != 200:
            return []

        data     = r.json()
        elements = data.get("element", [])

        scored = []
        for elem in elements:
            name  = elem.get("name", "")
            score = float(elem.get("score", {}).get("value", 0))
            if name:
                scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:6]]

    except Exception:
        return []


def _fetch_job_family(code, username, password):
    """
    Fetch job family for one occupation to determine Indian stream.
    Endpoint: GET /online/occupations/{code}
    Returns job family title string.
    """
    try:
        r = requests.get(
            f"{ONET_BASE_URL}/online/occupations/{code}",
            auth=_auth(username, password),
            headers=_headers(),
            timeout=15
        )

        if r.status_code != 200:
            return ""

        data = r.json()

        # Try job_family first then career_cluster
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
        "Computer":          "Technology",
        "Mathematical":      "Technology",
        "Architecture":      "Engineering",
        "Engineering":       "Engineering",
        "Healthcare":        "Healthcare",
        "Life":              "Research & Science",
        "Physical":          "Research & Science",
        "Science":           "Research & Science",
        "Business":          "Business",
        "Financial":         "Finance",
        "Management":        "Management",
        "Legal":             "Legal Services",
        "Education":         "Education",
        "Training":          "Education",
        "Arts":              "Creative Arts",
        "Design":            "Creative Arts",
        "Media":             "Media & Publishing",
        "Communication":     "Media & Publishing",
        "Social Service":    "Social Service",
        "Community":         "Social Service",
        "Farming":           "Agriculture",
        "Forestry":          "Agriculture",
        "Construction":      "Construction",
        "Extraction":        "Construction",
        "Transportation":    "Logistics & Transport",
        "Food":              "Hospitality",
        "Sales":             "Sales",
        "Office":            "Administration",
        "Production":        "Manufacturing",
        "Installation":      "Maintenance & Repair",
        "Personal Care":     "Personal Services",
        "Protective":        "Public Safety",
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


def fetch_all_onet_careers(username, password, supabase):
    """
    Main function — fetches all O*NET occupations with full details.

    Flow:
    1. Check Supabase cache (valid 30 days) → return if fresh
    2. Fetch all occupation codes and titles from O*NET
    3. For each occupation fetch interests, skills, job family
    4. Map to Indian stream and sector
    5. Save to Supabase cache
    6. Return as list of career dicts

    Returns list of dicts with keys:
        onet_code, job_title, sector, 12th_stream,
        primary_riasec, secondary_riasec, core_skills, cached_at
    """
    # Check Supabase cache
    try:
        cached = supabase.table("onet_careers").select("*").execute()
        if cached.data and len(cached.data) > 100:
            first    = cached.data[0]
            cached_at = datetime.fromisoformat(
                first.get("cached_at", "2000-01-01T00:00:00+00:00")
                .replace("Z", "+00:00")
            )
            age_days = (datetime.now(timezone.utc) - cached_at).days
            if age_days < CACHE_DAYS:
                print(f"Loaded {len(cached.data)} careers from Supabase cache")
                return cached.data
    except Exception as e:
        print(f"Cache check error: {e}")

    print("Fetching all occupations from O*NET API...")

    # Step 1 — Get all occupation codes
    occupations = _fetch_all_occupations(username, password)
    print(f"Total occupations found: {len(occupations)}")

    if not occupations:
        print("No occupations returned — check O*NET credentials")
        return []

    # Step 2 — Fetch details for each occupation
    careers = []
    total   = len(occupations)

    for i, occ in enumerate(occupations):
        code  = occ["code"]
        title = occ["title"]

        if (i + 1) % 50 == 0:
            print(f"Processing {i+1}/{total}: {title}")

        # Fetch details with small delay to respect rate limits
        primary, secondary = _fetch_interests(code, username, password)
        skills             = _fetch_skills(code, username, password)
        family             = _fetch_job_family(code, username, password)

        stream = _map_stream(family, title)
        sector = _map_sector(family, title)

        career = {
            "onet_code":        code,
            "job_title":        title,
            "sector":           sector,
            "12th_stream":      stream,
            "primary_riasec":   primary,
            "secondary_riasec": secondary,
            "core_skills":      ", ".join(skills),
            "cached_at":        datetime.now(timezone.utc).isoformat()
        }
        careers.append(career)
        time.sleep(0.1)  # Rate limit: 10 requests/second max

    print(f"Fetched details for {len(careers)} careers")

    # Step 3 — Save to Supabase in batches
    try:
        # Clear old cache
        supabase.table("onet_careers").delete().neq("onet_code", "").execute()

        # Insert in batches of 50
        for i in range(0, len(careers), 50):
            batch = careers[i:i+50]
            supabase.table("onet_careers").insert(batch).execute()

        print(f"Saved {len(careers)} careers to Supabase")
    except Exception as e:
        print(f"Error saving to Supabase: {e}")

    return careers
