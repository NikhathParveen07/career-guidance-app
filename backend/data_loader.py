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


@st.cache_data(ttl=86400)
def load_careers():
    """
    Returns (df, pinecone_needs_rebuild).

    pinecone_needs_rebuild is True only when a fresh O*NET fetch just ran
    (i.e., cache miss AND O*NET returned data).

    NOTE: st.session_state must NOT be accessed inside @st.cache_data.
    The per-session dedup of the Pinecone rebuild is handled in main()
    via st.session_state["pinecone_rebuilt"].
    """
    fresh_fetch = False

    # Try O*NET API path first
    try:
        api_key  = st.secrets.get("ONET_API_KEY", None)
        supabase = load_supabase()

        if api_key:
            careers = fetch_all_onet_careers(api_key, supabase)
            if careers and len(careers) > 10:
                df = pd.DataFrame(careers)
                if "12th_stream" in df.columns:
                    df = df.rename(columns={"12th_stream": "stream"})
                if "onet_code" in df.columns and "nco_id" not in df.columns:
                    df["nco_id"] = df["onet_code"]

                # fresh_fetch marks that this function body actually ran
                # (only happens on cache miss = genuine fresh O*NET pull)
                fresh_fetch = True
                df = _load_india_specific(df)
                return df, fresh_fetch

    except Exception as e:
        print(f"O*NET path failed, falling back to CSV: {e}")

    # Fallback: load from bundled CSV
    csv_path = os.path.join(DATA_DIR, "career_master_final.csv")
    if not os.path.exists(csv_path):
        st.error(f"Career data file not found at {csv_path}. Check your data directory.")
        return pd.DataFrame(), False

    df = pd.read_csv(csv_path)
    if "12th_stream" in df.columns:
        df = df.rename(columns={"12th_stream": "stream"})

    df = _load_india_specific(df)
    return df, fresh_fetch  # fresh_fetch is False for CSV path


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
