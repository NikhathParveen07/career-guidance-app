# ============================================
# backend/onet_loader.py
# Fetches all occupations from O*NET API
# Maps RIASEC, skills, sector, and Indian stream
# Caches in Supabase for 30 days
# ============================================
import requests
import json
from datetime import datetime, timezone
from requests.auth import HTTPBasicAuth


ONET_BASE_URL = "https://services.onetcenter.org/ws"
CACHE_DAYS    = 30


# ── Stream mapping from O*NET industry to Indian Class 12 stream ──
SECTOR_TO_STREAM = {
    # Science stream
    "Healthcare":                    "Science",
    "Healthcare Support":            "Science",
    "Healthcare Practitioners":      "Science",
    "Life, Physical, and Social Science": "Science",
    "Architecture and Engineering":  "Science",
    "Computer and Mathematical":     "Science",
    "Farming, Fishing, and Forestry":"Science",
    "Construction and Extraction":   "Science",
    "Installation, Maintenance, and Repair": "Science",
    "Production":                    "Science",

    # Commerce stream
    "Business and Financial Operations": "Commerce",
    "Management":                    "Commerce",
    "Sales and Related":             "Commerce",
    "Office and Administrative Support": "Commerce",
    "Legal":                         "Commerce",

    # Arts stream
    "Arts, Design, Entertainment, Sports, and Media": "Arts",
    "Education, Training, and Library": "Arts",
    "Community and Social Service":  "Arts",
    "Protective Service":            "Arts",
    "Personal Care and Service":     "Arts",

    # Vocational stream
    "Transportation and Material Moving": "Vocational",
    "Food Preparation and Serving Related": "Vocational",
    "Building and Grounds Cleaning and Maintenance": "Vocational",
    "Military Specific":             "Vocational",
}

# ── RIASEC code mapping from O*NET interest name ─────────────────
INTEREST_TO_RIASEC = {
    "Realistic":       "R",
    "Investigative":   "I",
    "Artistic":        "A",
    "Social":          "S",
    "Enterprising":    "E",
    "Conventional":    "C",
}


def _get_auth(username, password):
    return HTTPBasicAuth(username, password)


def _fetch_occupation_list(username, password):
    """
    Fetch list of all O*NET occupations.
    Returns list of {code, title} dicts.
    """
    try:
        all_occupations = []
        start = 1
        end   = 100

        while True:
            r = requests.get(
                f"{ONET_BASE_URL}/online/occupations",
                auth=_get_auth(username, password),
                params={"start": start, "end": end},
                headers={"Accept": "application/json"},
                timeout=30
            )
            if r.status_code != 200:
                break

            data         = r.json()
            occupations  = data.get("occupation", [])

            if not occupations:
                break

            for occ in occupations:
                all_occupations.append({
                    "code":  occ.get("code", ""),
                    "title": occ.get("title", "")
                })

            total = data.get("total", 0)
            if end >= total:
                break

            start += 100
            end   += 100

        return all_occupations

    except Exception as e:
        print(f"Error fetching occupation list: {e}")
        return []


def _fetch_occupation_details(code, username, password):
    """
    Fetch skills, interests (RIASEC), and category for one occupation.
    Returns dict with skills list and RIASEC codes.
    """
    result = {
        "skills":           [],
        "primary_riasec":   "R",
        "secondary_riasec": "I",
        "category":         "",
        "onet_code":        code
    }

    try:
        # Fetch interests (RIASEC)
        r_interests = requests.get(
            f"{ONET_BASE_URL}/online/occupations/{code}/details/interests",
            auth=_get_auth(username, password),
            headers={"Accept": "application/json"},
            timeout=15
        )

        if r_interests.status_code == 200:
            interests_data = r_interests.json()
            element_list   = (interests_data
                              .get("element", []))

            # Sort by score descending
            scored = []
            for elem in element_list:
                name  = elem.get("name", "")
                score = elem.get("score", {}).get("value", 0)
                if name in INTEREST_TO_RIASEC:
                    scored.append((INTEREST_TO_RIASEC[name], float(score)))

            scored.sort(key=lambda x: x[1], reverse=True)

            if len(scored) >= 1:
                result["primary_riasec"]   = scored[0][0]
            if len(scored) >= 2:
                result["secondary_riasec"] = scored[1][0]

        # Fetch skills
        r_skills = requests.get(
            f"{ONET_BASE_URL}/online/occupations/{code}/details/skills",
            auth=_get_auth(username, password),
            headers={"Accept": "application/json"},
            timeout=15
        )

        if r_skills.status_code == 200:
            skills_data  = r_skills.json()
            element_list = skills_data.get("element", [])

            # Take top 6 skills by importance score
            skill_scores = []
            for elem in element_list:
                name  = elem.get("name", "")
                score = elem.get("score", {}).get("value", 0)
                if name:
                    skill_scores.append((name, float(score)))

            skill_scores.sort(key=lambda x: x[1], reverse=True)
            result["skills"] = [s[0] for s in skill_scores[:6]]

        # Fetch occupation category
        r_occ = requests.get(
            f"{ONET_BASE_URL}/online/occupations/{code}",
            auth=_get_auth(username, password),
            headers={"Accept": "application/json"},
            timeout=15
        )

        if r_occ.status_code == 200:
            occ_data           = r_occ.json()
            result["category"] = occ_data.get(
                "career_cluster", {}).get("title", "") or \
                occ_data.get("job_family", {}).get("title", "") or ""

    except Exception as e:
        print(f"Error fetching details for {code}: {e}")

    return result


def _map_stream(category, title):
    """
    Map O*NET occupation category to Indian Class 12 stream.
    Falls back to keyword matching on title if category not found.
    """
    # Try direct category match
    for key, stream in SECTOR_TO_STREAM.items():
        if key.lower() in category.lower():
            return stream

    # Keyword fallback on title
    title_lower = title.lower()

    science_keywords = [
        "engineer", "scientist", "doctor", "nurse", "technician",
        "biologist", "chemist", "physicist", "analyst", "researcher",
        "medical", "health", "lab", "agricultural", "environmental",
        "software", "computer", "IT", "data", "cyber", "network"
    ]
    commerce_keywords = [
        "accountant", "financial", "banker", "manager", "business",
        "sales", "marketing", "economist", "auditor", "insurance",
        "broker", "consultant", "trader", "commerce", "tax", "legal",
        "lawyer", "attorney"
    ]
    arts_keywords = [
        "teacher", "writer", "journalist", "designer", "artist",
        "social worker", "counselor", "librarian", "curator",
        "translator", "psychologist", "educator", "director",
        "presenter", "media", "filmmaker"
    ]

    for kw in science_keywords:
        if kw in title_lower:
            return "Science"
    for kw in commerce_keywords:
        if kw in title_lower:
            return "Commerce"
    for kw in arts_keywords:
        if kw in title_lower:
            return "Arts"

    return "Vocational"


def _map_sector(category, title):
    """
    Map O*NET category to a readable sector name.
    """
    sector_map = {
        "Healthcare":           "Healthcare",
        "Technology":           "Technology",
        "Engineering":          "Engineering",
        "Science":              "Research & Science",
        "Business":             "Business",
        "Finance":              "Finance",
        "Legal":                "Legal Services",
        "Education":            "Education",
        "Arts":                 "Creative Arts",
        "Media":                "Media & Publishing",
        "Social":               "Social Service",
        "Agriculture":          "Agriculture",
        "Construction":         "Construction",
        "Transportation":       "Logistics & Transport",
        "Food":                 "Hospitality",
        "Sales":                "Sales",
        "Management":           "Management",
        "Computer":             "Technology",
        "Mathematical":         "Technology",
        "Architecture":         "Engineering",
    }

    for key, sector in sector_map.items():
        if key.lower() in category.lower():
            return sector

    # Fallback from title
    title_lower = title.lower()
    if any(w in title_lower for w in ["software", "data", "cyber", "network", "IT"]):
        return "Technology"
    if any(w in title_lower for w in ["doctor", "nurse", "medical", "health"]):
        return "Healthcare"
    if any(w in title_lower for w in ["teacher", "professor", "education"]):
        return "Education"
    if any(w in title_lower for w in ["accountant", "financial", "banker"]):
        return "Finance"

    return category if category else "General"


def fetch_all_onet_careers(username, password, supabase):
    """
    Main function — fetches all O*NET occupations with full details.

    Strategy:
    1. Check Supabase cache (valid 30 days)
    2. If stale or empty, fetch from O*NET API
    3. For each occupation fetch skills, RIASEC, category
    4. Map to Indian stream and sector
    5. Save to Supabase
    6. Return as list of career dicts

    Returns list of dicts with keys:
        onet_code, job_title, sector, 12th_stream,
        primary_riasec, secondary_riasec, core_skills
    """
    # Check Supabase cache
    try:
        cached = supabase.table("onet_careers").select("*").execute()
        if cached.data and len(cached.data) > 100:
            # Check if cache is fresh
            first_row  = cached.data[0]
            cached_at  = datetime.fromisoformat(
                first_row.get("cached_at", "2000-01-01"))
            age_days   = (datetime.now(timezone.utc) - cached_at).days
            if age_days < CACHE_DAYS:
                print(f"Loaded {len(cached.data)} careers from Supabase cache")
                return cached.data
    except Exception as e:
        print(f"Cache check error: {e}")

    print("Fetching careers from O*NET API...")

    # Fetch occupation list
    occupations = _fetch_occupation_list(username, password)
    print(f"Found {len(occupations)} occupations in O*NET")

    if not occupations:
        return []

    careers = []
    for i, occ in enumerate(occupations):
        code  = occ["code"]
        title = occ["title"]

        print(f"Fetching {i+1}/{len(occupations)}: {title}")

        # Fetch details
        details = _fetch_occupation_details(code, username, password)

        # Map stream and sector
        stream = _map_stream(details["category"], title)
        sector = _map_sector(details["category"], title)

        career = {
            "onet_code":        code,
            "job_title":        title,
            "sector":           sector,
            "12th_stream":      stream,
            "primary_riasec":   details["primary_riasec"],
            "secondary_riasec": details["secondary_riasec"],
            "core_skills":      ", ".join(details["skills"]),
            "cached_at":        datetime.now(timezone.utc).isoformat()
        }
        careers.append(career)

    # Save to Supabase
    try:
        # Delete old cache
        supabase.table("onet_careers").delete().neq("onet_code", "").execute()

        # Insert in batches of 50
        batch_size = 50
        for i in range(0, len(careers), batch_size):
            batch = careers[i:i+batch_size]
            supabase.table("onet_careers").insert(batch).execute()

        print(f"Saved {len(careers)} careers to Supabase")
    except Exception as e:
        print(f"Error saving to Supabase: {e}")

    return careers
