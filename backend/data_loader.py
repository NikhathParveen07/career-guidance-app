# ============================================
# backend/data_loader.py
# ============================================
import os
import pandas as pd
import streamlit as st
from pinecone import Pinecone
from supabase import create_client
from sentence_transformers import SentenceTransformer

from backend.onet_loader import fetch_all_onet_careers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


@st.cache_resource
def load_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )


def _load_india_specific(existing_df):
    """Merge India-specific careers into an existing DataFrame."""
    india_path = os.path.join(DATA_DIR, "india_specific_careers.csv")
    if not os.path.exists(india_path):
        return existing_df

    try:
        india_df = pd.read_csv(india_path)
        if "12th_stream" in india_df.columns:
            india_df = india_df.rename(columns={"12th_stream": "stream"})
        shared_cols = [c for c in existing_df.columns if c in india_df.columns]
        india_df = india_df[shared_cols]
        merged = pd.concat([existing_df, india_df], ignore_index=True).drop_duplicates(
            subset="job_title", keep="first"
        ).reset_index(drop=True)
        print(f"Career list after India merge: {len(merged)} careers")
        return merged
    except Exception as e:
        print(f"Failed to merge India-specific careers: {e}")
        return existing_df


def _load_from_supabase_cache(supabase):
    """
    Fast path: load careers directly from Supabase onet_careers table.
    Returns DataFrame or None if cache is empty / unavailable.
    This avoids triggering a slow O*NET API re-fetch on every cold start.
    """
    try:
        from datetime import datetime, timezone

        cached = supabase.table("onet_careers").select("*").execute()
        if not cached.data or len(cached.data) < 100:
            return None

        # Check freshness
        first = cached.data[0]
        cached_str = first.get("cached_at", "2000-01-01T00:00:00+00:00")
        cached_str = cached_str.replace("Z", "+00:00")
        cached_at = datetime.fromisoformat(cached_str)
        age_days = (datetime.now(timezone.utc) - cached_at).days

        if age_days >= 30:
            print(f"Supabase cache is {age_days} days old — will use CSV fallback")
            return None

        df = pd.DataFrame(cached.data)
        print(f"Loaded {len(df)} careers from Supabase cache (age: {age_days} days)")
        return df

    except Exception as e:
        print(f"Supabase cache load failed: {e}")
        return None


def _load_from_csv():
    """
    Load careers from bundled CSV files.
    Tries onet_careers_filtered.csv first, then onet_careers_rows.csv.
    Returns DataFrame or None if no file found.
    """
    # Try primary filtered CSV
    for filename in ["onet_careers_filtered.csv", "onet_careers_rows.csv"]:
        csv_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                print(f"Loaded {len(df)} careers from {filename}")
                return df
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    print(f"No CSV career file found in {DATA_DIR}")
    return None


@st.cache_data(ttl=86400)
def load_careers():
    """
    Returns (df, pinecone_needs_rebuild).

    Loading priority:
      1. Supabase onet_careers cache (fast, skips O*NET API)
      2. Bundled CSV (onet_careers_filtered.csv or onet_careers_rows.csv)
      3. O*NET API live fetch (slow — only if both above are unavailable)

    pinecone_needs_rebuild is True only when a fresh O*NET fetch just ran.

    NOTE: st.session_state must NOT be accessed inside @st.cache_data.
    The per-session dedup of the Pinecone rebuild is handled in main()
    via st.session_state["pinecone_rebuilt"].
    """
    supabase = load_supabase()

    # ── Fast path 1: Supabase cache ───────────────────────────
    df = _load_from_supabase_cache(supabase)
    if df is not None:
        if "12th_stream" in df.columns:
            df = df.rename(columns={"12th_stream": "stream"})
        if "onet_code" in df.columns and "nco_id" not in df.columns:
            df["nco_id"] = df["onet_code"]
        df = _load_india_specific(df)
        return df, False  # no Pinecone rebuild needed

    # ── Fast path 2: Bundled CSV ──────────────────────────────
    df = _load_from_csv()
    if df is not None:
        if "12th_stream" in df.columns:
            df = df.rename(columns={"12th_stream": "stream"})
        if "onet_code" in df.columns and "nco_id" not in df.columns:
            df["nco_id"] = df["onet_code"]
        df = _load_india_specific(df)
        return df, False

    # ── Slow path: O*NET live API fetch ───────────────────────
    # Only reached if BOTH Supabase and CSV are unavailable.
    # This is the path that causes the long hang — only runs as last resort.
    print("WARNING: Falling back to live O*NET API fetch — this will be slow")
    try:
        api_key = st.secrets.get("ONET_API_KEY", None)
        if api_key:
            careers = fetch_all_onet_careers(api_key, supabase)
            if careers and len(careers) > 10:
                df = pd.DataFrame(careers)
                if "12th_stream" in df.columns:
                    df = df.rename(columns={"12th_stream": "stream"})
                if "onet_code" in df.columns and "nco_id" not in df.columns:
                    df["nco_id"] = df["onet_code"]
                df = _load_india_specific(df)
                return df, True  # Pinecone rebuild needed after fresh fetch
    except Exception as e:
        print(f"O*NET API fetch failed: {e}")

    st.error(
        "Career data could not be loaded. "
        "Expected one of these files in the data/ directory: "
        "onet_careers_filtered.csv or onet_careers_rows.csv"
    )
    return pd.DataFrame(), False


@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_pinecone_index():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc.Index("career-discovery")


@st.cache_data
def load_students():
    path = os.path.join(DATA_DIR, "students.csv")
    return pd.read_csv(path)


@st.cache_data
def load_interactions():
    path = os.path.join(DATA_DIR, "interactions.csv")
    return pd.read_csv(path)
