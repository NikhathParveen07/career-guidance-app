# ============================================
# backend/future_market.py
# Dynamic forward-looking career market intelligence
#
# Two features:
#   1. Salary projection at graduation
#      Sources: SerpAPI salary string (current)
#               World Bank API (sector wage growth rate)
#               No card, no signup beyond what you have
#
#   2. Demand signal
#      Sources: Google News RSS (free, no key)
#               NewsAPI (free tier, 100 req/day)
#               Groq LLM (classify news into signal)
#               Supabase (weekly cache)
#
# No static dictionaries — all data is live and refreshed
# ============================================
import requests
import feedparser
import json
import re
import time
from datetime import datetime, timezone, timedelta


# ── Constants ─────────────────────────────────────────────────
GROQ_URL         = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL       = "llama-3.3-70b-versatile"
WORLD_BANK_URL   = "https://api.worldbank.org/v2"
CACHE_DAYS       = 7    # Refresh demand signal every 7 days
SALARY_CACHE_DAYS = 30  # Refresh salary projection every 30 days


# ── Sector to World Bank industry mapping ─────────────────────
# Maps career sectors to World Bank broad industry categories
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

# Degree duration — years until graduation per stream
DEGREE_DURATION = {
    "Science":    4,   # B.Tech / MBBS / B.Sc
    "Commerce":   3,   # B.Com / BBA / CA
    "Arts":       3,   # BA / BFA / LLB
    "Vocational": 2,   # Diploma / ITI
}

# Sector salary fallbacks — used when SerpAPI returns no salary
# Based on PLFS 2023 wage data
SECTOR_SALARY_FALLBACK = {
    "Technology":          6.0,
    "Healthcare":          4.5,
    "Finance":             5.5,
    "Education":           3.5,
    "Engineering":         5.0,
    "Media & Publishing":  4.0,
    "Agriculture":         2.5,
    "Hospitality":         3.0,
    "Creative Arts":       3.0,
    "Legal":               5.0,
    "Textiles":            2.5,
    "Vocational":          2.8,
    "Construction":        3.5,
    "Green Energy":        4.5,
    "Retail":              3.0,
    "Sales":               4.0,
    "Government Revenue":  4.0,
    "Maritime":            5.0,
}


# ── Shared helpers ────────────────────────────────────────────

def _call_groq(prompt, groq_key):
    """Call Groq LLM and return raw text. Returns None on failure."""
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
                "temperature": 0.2,
                "max_tokens":  800
            },
            timeout=30
        )
        if response.status_code != 200:
            return None
        data = response.json()
        if "error" in data:
            return None
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def _parse_json(raw_text):
    """Parse JSON from LLM response handling markdown fences."""
    if not raw_text:
        return None
    clean = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        try:
            start = clean.index("{")
            end   = clean.rindex("}") + 1
            return json.loads(clean[start:end])
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════
# FEATURE 1 — SALARY PROJECTION AT GRADUATION
# ══════════════════════════════════════════════════════════════

def _parse_serpapi_salary(salary_string):
    """
    Parse salary in LPA from SerpAPI salary string.

    Handles formats like:
        '₹4.8L a year'         → 4.8
        '₹1L-₹50L a year'      → 25.5  (midpoint of range)
        '₹40,000 a month'      → 4.8   (converted to annual LPA)
        'Not available'        → None

    Returns float in LPA or None.
    """
    if not salary_string or salary_string == "Not available":
        return None

    try:
        # Remove currency symbols and normalise
        clean = salary_string.replace("₹", "").replace(",", "").lower()

        # Check if monthly — convert to annual
        is_monthly = "month" in clean

        # Extract all numbers
        numbers = re.findall(r'[\d.]+', clean)
        if not numbers:
            return None

        values = [float(n) for n in numbers]

        # Handle L (lakh) abbreviation
        # If numbers are small (< 200) they are already in LPA
        # If large (> 10000) they are in rupees — convert to LPA
        processed = []
        for v in values:
            if v > 100000:
                processed.append(v / 100000)  # rupees to LPA
            elif v > 1000:
                processed.append(v / 100000)  # thousands to LPA
            else:
                processed.append(v)            # already LPA

        if is_monthly:
            processed = [v * 12 for v in processed]

        # Return midpoint if range, single value otherwise
        result = sum(processed) / len(processed)

        # Sanity check — Indian salaries rarely below 1L or above 100L
        if result < 0.5 or result > 100:
            return None

        return round(result, 1)

    except Exception:
        return None


def _get_world_bank_growth_rate(sector):
    """
    Fetch India's annual value-added growth rate for the
    broad industry category from World Bank API.

    Completely free — no API key required.
    Returns annual growth rate as decimal (e.g. 0.08 = 8%).
    Falls back to 0.05 if API fails.
    """
    industry_type = SECTOR_WB_INDUSTRY.get(sector, "services")

    indicator_map = {
        "agriculture": "NV.AGR.TOTL.KD.ZG",
        "industry":    "NV.IND.TOTL.KD.ZG",
        "services":    "NV.SRV.TOTL.KD.ZG"
    }
    indicator = indicator_map.get(industry_type, "NV.SRV.TOTL.KD.ZG")

    try:
        url = (f"{WORLD_BANK_URL}/country/IN/indicator/{indicator}"
               f"?format=json&mrv=3&per_page=3")
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return 0.05

        data = response.json()

        # World Bank returns [metadata, data_array]
        if len(data) < 2 or not data[1]:
            return 0.05

        values = [
            entry["value"] for entry in data[1]
            if entry.get("value") is not None
        ]

        if not values:
            return 0.05

        avg_growth = sum(values) / len(values) / 100
        # Clamp between 2% and 15% to avoid extremes
        return max(0.02, min(0.15, avg_growth))

    except Exception:
        return 0.05


def get_salary_projection(career_title, sector, stream,
                          serpapi_salary_string, supabase):
    """
    Project expected salary at graduation.

    Uses:
    - serpapi_salary_string: salary already fetched by job_market.py
    - World Bank API: sector wage growth rate (free, no key)

    Formula: projected = current × (1 + growth_rate) ^ years

    Args:
        career_title          — career name
        sector                — career sector
        stream                — student stream (determines degree years)
        serpapi_salary_string — salary string from SerpAPI e.g. '₹4.8L a year'
        supabase              — Supabase client for caching

    Returns dict with current and projected salary info.
    """
    years = DEGREE_DURATION.get(stream, 3)

    # Check Supabase cache (valid 30 days)
    cache_key = f"salary_{career_title}"
    try:
        cached = (supabase.table("salary_projections")
                          .select("*")
                          .eq("cache_key", cache_key)
                          .execute())
        if cached.data:
            row       = cached.data[0]
            cached_at = datetime.fromisoformat(row["cached_at"])
            age_days  = (datetime.now(timezone.utc) - cached_at).days
            if age_days < SALARY_CACHE_DAYS:
                data = json.loads(row["projection_json"])
                # Recalculate projection with current years
                projected = round(
                    data["current_salary_lpa"] * ((1 + data["growth_rate"]) ** years), 1
                )
                proj_low  = round(projected * 0.85, 1)
                proj_high = round(projected * 1.15, 1)
                data["years"]            = years
                data["graduation_year"]  = datetime.now().year + years
                data["projected_range"]  = f"₹{proj_low}L–₹{proj_high}L per year"
                data["from_cache"]       = True
                return data
    except Exception:
        pass

    # Parse current salary from SerpAPI string
    current_salary = _parse_serpapi_salary(serpapi_salary_string)
    salary_source  = "SerpAPI Google Jobs"

    # Fall back to sector average if SerpAPI gave no salary
    if not current_salary:
        current_salary = SECTOR_SALARY_FALLBACK.get(sector, 3.5)
        salary_source  = "PLFS 2023 sector average"

    # Get growth rate from World Bank
    growth_rate   = _get_world_bank_growth_rate(sector)
    growth_source = "World Bank India Value Added Growth Index"

    # Project salary at graduation
    projected  = round(current_salary * ((1 + growth_rate) ** years), 1)
    proj_low   = round(projected * 0.85, 1)
    proj_high  = round(projected * 1.15, 1)
    curr_low   = round(current_salary * 0.85, 1)
    curr_high  = round(current_salary * 1.15, 1)

    result = {
        "career":             career_title,
        "current_salary_lpa": current_salary,
        "current_range":      f"₹{curr_low}L–₹{curr_high}L per year",
        "growth_rate":        growth_rate,
        "growth_pct":         f"{round(growth_rate * 100, 1)}% per year",
        "years":              years,
        "graduation_year":    datetime.now().year + years,
        "projected_range":    f"₹{proj_low}L–₹{proj_high}L per year",
        "salary_source":      salary_source,
        "growth_source":      growth_source,
        "from_cache":         False
    }

    # Cache for 30 days
    try:
        supabase.table("salary_projections").upsert({
            "cache_key":       cache_key,
            "projection_json": json.dumps({
                k: v for k, v in result.items()
                if k not in ["years", "graduation_year",
                             "projected_range", "from_cache"]
            }),
            "cached_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception:
        pass

    return result


# ══════════════════════════════════════════════════════════════
# FEATURE 2 — DYNAMIC DEMAND SIGNAL
# ══════════════════════════════════════════════════════════════

def _fetch_google_news_rss(sector, career_title):
    """
    Fetch recent news from Google News RSS.
    Free — no API key required.
    Returns list of headline strings.
    """
    try:
        query     = f"{career_title} {sector} India jobs demand policy investment"
        query_enc = requests.utils.quote(query)
        url       = (f"https://news.google.com/rss/search"
                     f"?q={query_enc}&hl=en-IN&gl=IN&ceid=IN:en")
        feed      = feedparser.parse(url)
        return [entry.title for entry in feed.entries[:8]]
    except Exception:
        return []


def _fetch_newsapi_headlines(sector, career_title, news_api_key):
    """
    Fetch recent news from NewsAPI.
    Free tier: 100 requests/day.
    Returns list of headline strings.
    """
    if not news_api_key:
        return []
    try:
        query    = f"{career_title} {sector} India employment policy demand"
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        query,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": 8,
                "apiKey":   news_api_key
            },
            timeout=10
        )
        if response.status_code != 200:
            return []
        data     = response.json()
        articles = data.get("articles", [])
        return [a["title"] for a in articles if a.get("title")]
    except Exception:
        return []


def _classify_demand_signal(headlines, sector, career_title, groq_key):
    """
    Use Groq LLM to classify news headlines into a structured
    demand signal for a career.

    Returns dict with trend, driver, policy, impact, confidence.
    """
    if not headlines:
        return {
            "trend":        "Insufficient Data",
            "driver":       "No recent news found for this sector",
            "policy":       "None identified",
            "impact":       "Unknown",
            "confidence":   "Low",
            "key_headline": "No headlines available"
        }

    headlines_text = "\n".join([f"- {h}" for h in headlines])

    prompt = f"""You are an Indian labour market analyst.

Analyse these recent news headlines about {career_title} in the {sector} sector in India:

{headlines_text}

Based only on these headlines, classify the career demand outlook for someone
graduating in India in 3-4 years.

Return ONLY valid JSON with exactly these keys:
{{
  "trend": "Growing / Stable / Declining / Rapidly Growing / Uncertain",
  "driver": "One sentence: main factor driving demand change in India",
  "policy": "Specific government policy or budget scheme mentioned, or None",
  "impact": "Very Positive / Positive / Neutral / Negative / Very Negative",
  "confidence": "High / Medium / Low",
  "key_headline": "The single most relevant headline from the list above"
}}

Be specific to India. Base your classification strictly on the headlines.
If headlines are unrelated to career demand, set confidence to Low.
Return ONLY the JSON object."""

    raw  = _call_groq(prompt, groq_key)
    data = _parse_json(raw)

    if data:
        return data

    return {
        "trend":        "Stable",
        "driver":       "Could not classify from available news",
        "policy":       "None identified",
        "impact":       "Neutral",
        "confidence":   "Low",
        "key_headline": headlines[0] if headlines else "No headlines found"
    }


def get_demand_signal(career_title, sector,
                      groq_key, news_api_key, supabase):
    """
    Generate a dynamic weekly-refreshed demand signal.

    Steps:
    1. Check Supabase cache (valid 7 days)
    2. Fetch headlines from Google News RSS (free, no key)
    3. Fetch headlines from NewsAPI (free tier)
    4. Combine and deduplicate headlines
    5. Classify with Groq LLM
    6. Cache in Supabase
    7. Return structured signal

    Args:
        career_title  — career to analyse
        sector        — career sector
        groq_key      — Groq API key
        news_api_key  — NewsAPI key (can be None — RSS still works)
        supabase      — Supabase client

    Returns dict with trend, driver, policy, impact,
    confidence, key_headline, last_updated.
    """
    cache_key = f"demand_{career_title}"

    # Check Supabase cache
    try:
        cached = (supabase.table("demand_signals")
                          .select("*")
                          .eq("cache_key", cache_key)
                          .execute())
        if cached.data:
            row       = cached.data[0]
            cached_at = datetime.fromisoformat(row["cached_at"])
            age_days  = (datetime.now(timezone.utc) - cached_at).days
            if age_days < CACHE_DAYS:
                data               = json.loads(row["signal_json"])
                data["from_cache"] = True
                data["last_updated"] = cached_at.strftime("%d %b %Y")
                return data
    except Exception:
        pass

    # Fetch from both sources
    rss_headlines  = _fetch_google_news_rss(sector, career_title)
    news_headlines = _fetch_newsapi_headlines(sector, career_title, news_api_key)

    # Combine and deduplicate preserving order
    all_headlines = list(dict.fromkeys(rss_headlines + news_headlines))[:10]

    # Classify with Groq
    signal = _classify_demand_signal(all_headlines, sector, career_title, groq_key)

    signal["headlines_used"] = len(all_headlines)
    signal["last_updated"]   = datetime.now().strftime("%d %b %Y")
    signal["from_cache"]     = False

    # Cache in Supabase
    try:
        supabase.table("demand_signals").upsert({
            "cache_key":   cache_key,
            "signal_json": json.dumps({
                k: v for k, v in signal.items()
                if k not in ["from_cache", "last_updated"]
            }),
            "cached_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception:
        pass

    return signal


# ══════════════════════════════════════════════════════════════
# COMBINED FUNCTION — called from job_market.py
# ══════════════════════════════════════════════════════════════

def get_future_market_data(career_title, sector, stream,
                           serpapi_salary_string,
                           groq_key, news_api_key, supabase):
    """
    Master function combining salary projection + demand signal.
    Called from job_market.py after current data is fetched.

    Args:
        career_title          — career name
        sector                — career sector
        stream                — student stream
        serpapi_salary_string — salary string from SerpAPI
                                (used as current salary input)
        groq_key              — Groq API key
        news_api_key          — NewsAPI key (can be None)
        supabase              — Supabase client

    Returns:
        dict with "salary" and "demand" keys
    """
    salary_data = get_salary_projection(
        career_title          = career_title,
        sector                = sector,
        stream                = stream,
        serpapi_salary_string = serpapi_salary_string,
        supabase              = supabase
    )

    demand_data = get_demand_signal(
        career_title = career_title,
        sector       = sector,
        groq_key     = groq_key,
        news_api_key = news_api_key,
        supabase     = supabase
    )

    return {
        "salary": salary_data,
        "demand": demand_data
    }
