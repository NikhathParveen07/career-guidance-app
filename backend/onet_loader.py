# ============================================
# backend/onet_loader.py
# ============================================
import requests
import time
from datetime import datetime, timezone
from backend.onet_india_filter import is_india_relevant
from backend.embed_careers import embed_and_upsert


ONET_BASE_URL = "https://api-v2.onetcenter.org"
CACHE_DAYS    = 30


SOC_TO_STREAM = {
    "11": "Commerce",
    "13": "Commerce",
    "15": "Science",
    "17": "Science",
    "19": "Science",
    "21": "Arts",
    "23": "Commerce",
    "25": "Arts",
    "27": "Arts",
    "29": "Science",
    "31": "Science",
    "33": "Vocational",
    "35": "Vocational",
    "37": "Vocational",
    "39": "Vocational",
    "41": "Commerce",
    "43": "Commerce",
    "45": "Science",
    "47": "Vocational",
    "49": "Vocational",
    "51": "Vocational",
    "53": "Vocational",
    "55": "Vocational",
}

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
    return {
        "Accept":    "application/json",
        "X-API-Key": api_key
    }


def _get_soc_prefix(code):
    return code.split("-")[0] if "-" in code else "11"


def _fetch_all_occupations(api_key):
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

    # Apply India relevance filter
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

        primary, secondary = _fetch_interests(code, api_key)
        skills             = _fetch_skills(code, api_key)

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


def rebuild_pinecone_after_refresh(df, pinecone_index, sentence_model):
    """
    Wipe and rebuild the entire Pinecone index from the merged DataFrame.
    Called from app.py after a fresh O*NET fetch cycle.
    """
    try:
        pinecone_index.delete(delete_all=True)
        print("Pinecone index cleared")
    except Exception as e:
        print(f"Could not clear Pinecone index: {e}")

    embed_and_upsert(df, pinecone_index, sentence_model)
