# ============================================
# backend/future_market.py
# Dynamic forward-looking career market intelligence
# ============================================
import requests
import feedparser
import json
import re
from datetime import datetime, timezone

GROQ_URL          = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL        = "llama-3.3-70b-versatile"
WORLD_BANK_URL    = "https://api.worldbank.org/v2"
CACHE_DAYS        = 7
SALARY_CACHE_DAYS = 30

SECTOR_WB_INDUSTRY = {
    "Technology":          "industry",
    "Healthcare":          "services",
    "Finance":             "services",
    "Education":           "services",
    "Media & Publishing":  "services",
    "Engineering":         "industry",
    "Construction":        "industry",
    "Agriculture":         "agriculture",
    "Hospitality":         "services",
    "Green Energy":        "industry",
    "Creative Arts":       "services",
    "Legal":               "services",
    "Textiles":            "industry",
    "Vocational":          "industry",
    "Sales":               "services",
    "Government Revenue":  "services",
    "Maritime":            "industry",
    "Retail":              "services",
}

DEGREE_DURATION = {
    "Science":    4,
    "Commerce":   3,
    "Arts":       3,
    "Vocational": 2,
}

SECTOR_SALARY_FALLBACK = {
    "Technology": 6.0, "Healthcare": 4.5, "Finance": 5.5,
    "Education": 3.5, "Engineering": 5.0, "Media & Publishing": 4.0,
    "Agriculture": 2.5, "Hospitality": 3.0, "Creative Arts": 3.0,
    "Legal": 5.0, "Textiles": 2.5, "Vocational": 2.8,
    "Construction": 3.5, "Green Energy": 4.5, "Retail": 3.0,
    "Sales": 4.0, "Government Revenue": 4.0, "Maritime": 5.0,
}


# ── Helpers ───────────────────────────────────────────────────

def _call_groq(prompt, groq_key):
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {groq_key}",
                     "Content-Type": "application/json"},
            json={"model": GROQ_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.2, "max_tokens": 1000},
            timeout=30
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if "error" in data:
            return None
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def _parse_json(raw):
    if not raw:
        return None
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        try:
            return json.loads(clean[clean.index("{"):clean.rindex("}")+1])
        except Exception:
            return None


def _parse_serpapi_salary(salary_string):
    if not salary_string or salary_string == "Not available":
        return None
    try:
        clean      = salary_string.replace("₹", "").replace(",", "").lower()
        is_monthly = "month" in clean
        numbers    = re.findall(r'[\d.]+', clean)
        if not numbers:
            return None
        values = []
        for n in numbers:
            v = float(n)
            if v > 100000:
                v = v / 100000
            elif v > 1000:
                v = v / 100000
            values.append(v * 12 if is_monthly else v)
        result = sum(values) / len(values)
        return round(result, 1) if 0.5 <= result <= 100 else None
    except Exception:
        return None


def _get_world_bank_growth_rate(sector):
    industry = SECTOR_WB_INDUSTRY.get(sector, "services")
    ind_map  = {
        "agriculture": "NV.AGR.TOTL.KD.ZG",
        "industry":    "NV.IND.TOTL.KD.ZG",
        "services":    "NV.SRV.TOTL.KD.ZG"
    }
    try:
        r = requests.get(
            f"{WORLD_BANK_URL}/country/IN/indicator/{ind_map[industry]}"
            f"?format=json&mrv=3&per_page=3",
            timeout=10
        )
        if r.status_code != 200:
            return 0.05
        data   = r.json()
        values = [e["value"] for e in data[1] if e.get("value") is not None]
        if not values:
            return 0.05
        return max(0.02, min(0.15, sum(values) / len(values) / 100))
    except Exception:
        return 0.05


def _fetch_headlines(sector, career_title, news_api_key):
    headlines = []
    try:
        q   = requests.utils.quote(f"{career_title} {sector} India jobs demand policy")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        headlines += [e.title for e in feedparser.parse(url).entries[:6]]
    except Exception:
        pass
    if news_api_key:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": f"{career_title} {sector} India employment",
                        "language": "en", "sortBy": "publishedAt",
                        "pageSize": 6, "apiKey": news_api_key},
                timeout=10
            )
            if r.status_code == 200:
                headlines += [a["title"] for a in r.json().get("articles", [])
                              if a.get("title")]
        except Exception:
            pass
    return list(dict.fromkeys(headlines))[:10]


def _generate_market_intelligence(career_title, sector, headlines, groq_key):
    headlines_text = "\n".join([f"- {h}" for h in headlines]) if headlines \
                     else "No recent headlines available."

    prompt = f"""You are an Indian labour market analyst helping Class 12 students
understand career prospects.

Career: {career_title}
Sector: {sector}
Recent news headlines:
{headlines_text}

Generate a career market intelligence report. Return ONLY valid JSON:
{{
  "outlook": {{
    "trend": "Rapidly Growing / Growing / Stable / Declining / Uncertain",
    "government_backed": true or false,
    "explanation": "One clear sentence a 12th grader would understand"
  }},
  "competition": {{
    "level": "Low / Moderate / Competitive / Very Competitive",
    "explanation": "One sentence about graduates produced vs jobs available in India"
  }},
  "policy": {{
    "exists": true or false,
    "scheme_name": "Name of government scheme or null",
    "explanation": "One sentence explaining how this policy affects this career, or null"
  }},
  "source_outlook": "NSDC [Sector] Sector Report 2024 or relevant source",
  "source_policy": "Union Budget 2024-25 or relevant source or null"
}}

Be specific to India. Use simple language. Return ONLY the JSON object."""

    raw  = _call_groq(prompt, groq_key)
    data = _parse_json(raw)

    if data:
        return data

    return {
        "outlook": {
            "trend":             "Stable",
            "government_backed": False,
            "explanation":       "Could not retrieve current market data."
        },
        "competition": {
            "level":       "Moderate",
            "explanation": "Competition level data unavailable at this time."
        },
        "policy": {
            "exists":      False,
            "scheme_name": None,
            "explanation": None
        },
        "source_outlook": None,
        "source_policy":  None
    }


def _get_salary_data(career_title, sector, stream,
                     serpapi_salary_string, supabase):
    years     = DEGREE_DURATION.get(stream, 3)
    cache_key = f"salary_{career_title}"

    try:
        cached = (supabase.table("salary_projections")
                          .select("*").eq("cache_key", cache_key).execute())
        if cached.data:
            row      = cached.data[0]
            age_days = (datetime.now(timezone.utc) -
                        datetime.fromisoformat(row["cached_at"])).days
            if age_days < SALARY_CACHE_DAYS:
                d         = json.loads(row["projection_json"])
                projected = round(d["current_lpa"] * ((1 + d["growth_rate"]) ** years), 1)
                mid       = round(projected * 2.2, 1)
                return {
                    **d,
                    "years":           years,
                    "graduation_year": datetime.now().year + years,
                    "projected_low":   round(projected * 0.85, 1),
                    "projected_high":  round(projected * 1.15, 1),
                    "mid_low":         round(mid * 0.85, 1),
                    "mid_high":        round(mid * 1.15, 1),
                    "from_cache":      True
                }
    except Exception:
        pass

    current_lpa   = _parse_serpapi_salary(serpapi_salary_string)
    salary_source = "SerpAPI live job listings"
    if not current_lpa:
        current_lpa   = SECTOR_SALARY_FALLBACK.get(sector, 3.5)
        salary_source = "PLFS 2023 sector average"

    growth_rate = _get_world_bank_growth_rate(sector)
    projected   = round(current_lpa * ((1 + growth_rate) ** years), 1)
    mid         = round(projected * 2.2, 1)

    result = {
        "current_lpa":     current_lpa,
        "current_low":     round(current_lpa * 0.85, 1),
        "current_high":    round(current_lpa * 1.15, 1),
        "growth_rate":     growth_rate,
        "growth_pct":      f"{round(growth_rate * 100, 1)}% per year",
        "years":           years,
        "graduation_year": datetime.now().year + years,
        "projected_low":   round(projected * 0.85, 1),
        "projected_high":  round(projected * 1.15, 1),
        "mid_low":         round(mid * 0.85, 1),
        "mid_high":        round(mid * 1.15, 1),
        "salary_source":   salary_source,
        "growth_source":   "World Bank India Value Added Growth Index",
        "from_cache":      False
    }

    try:
        supabase.table("salary_projections").upsert({
            "cache_key":       cache_key,
            "projection_json": json.dumps({
                k: v for k, v in result.items()
                if k not in ["years", "graduation_year", "projected_low",
                             "projected_high", "mid_low", "mid_high", "from_cache"]
            }),
            "cached_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception:
        pass

    return result


# ── Master function ───────────────────────────────────────────

def get_future_market_data(career_title, sector, stream,
                           serpapi_salary_string,
                           serpapi_companies,
                           serpapi_locations,
                           groq_key, news_api_key, supabase):

    salary = _get_salary_data(
        career_title, sector, stream,
        serpapi_salary_string, supabase
    )

    cache_key    = f"intel_{career_title}"
    intelligence = None

    try:
        cached = (supabase.table("demand_signals")
                          .select("*").eq("cache_key", cache_key).execute())
        if cached.data:
            row      = cached.data[0]
            age_days = (datetime.now(timezone.utc) -
                        datetime.fromisoformat(row["cached_at"])).days
            if age_days < CACHE_DAYS:
                intelligence                 = json.loads(row["signal_json"])
                intelligence["from_cache"]   = True
                intelligence["last_updated"] = datetime.fromisoformat(
                    row["cached_at"]).strftime("%d %b %Y")
    except Exception:
        pass

    if not intelligence:
        headlines    = _fetch_headlines(sector, career_title, news_api_key)
        intelligence = _generate_market_intelligence(
            career_title, sector, headlines, groq_key
        )
        intelligence["headlines_used"] = len(headlines)
        intelligence["last_updated"]   = datetime.now().strftime("%d %b %Y")
        intelligence["from_cache"]     = False

        try:
            supabase.table("demand_signals").upsert({
                "cache_key":   cache_key,
                "signal_json": json.dumps({
                    k: v for k, v in intelligence.items()
                    if k not in ["from_cache", "last_updated"]
                }),
                "cached_at": datetime.now(timezone.utc).isoformat()
            }).execute()
        except Exception:
            pass

    return {
        "salary":       salary,
        "intelligence": intelligence,
        "companies":    serpapi_companies,
        "cities":       serpapi_locations
    }
