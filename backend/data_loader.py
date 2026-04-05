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


@st.cache_data(ttl=86400)
def load_careers():
    """
    Returns (df, pinecone_needs_rebuild).
    pinecone_needs_rebuild is True only when a fresh O*NET fetch just ran.

    IMPORTANT: st.session_state must NOT be accessed inside @st.cache_data.
    Cache functions are shared across sessions; session state is per-user.
    The fresh_fetch flag is set to True whenever this function body executes
    (which only happens on a cache miss, i.e., a genuine fresh API fetch).
    Per-session dedup of the Pinecone rebuild is handled in main() via
    st.session_state["pinecone_rebuilt"].
    """
    fresh_fetch = False

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

                # fresh_fetch = True whenever this function body runs,
                # because @st.cache_data means it only executes on cache miss
                # (i.e., first run or after TTL expiry — a genuine fresh fetch).
                fresh_fetch = True

                # Merge India-specific careers
                india_path = os.path.join(DATA_DIR, "india_specific_careers.csv")
                if os.path.exists(india_path):
                    india_df = pd.read_csv(india_path)
                    if "12th_stream" in india_df.columns:
                        india_df = india_df.rename(columns={"12th_stream": "stream"})
                    shared_cols = [c for c in df.columns if c in india_df.columns]
                    india_df = india_df[shared_cols]
                    df = pd.concat([df, india_df], ignore_index=True).drop_duplicates(
                        subset="job_title", keep="first"
                    )
                    print(f"Career list after merge: {len(df)} careers")

                return df, fresh_fetch

    except Exception as e:
        st.warning(f"O*NET unavailable, loading from CSV: {e}")

    # Fallback to CSV
    csv_path = os.path.join(DATA_DIR, "career_master_final.csv")
    df = pd.read_csv(csv_path)
    if "12th_stream" in df.columns:
        df = df.rename(columns={"12th_stream": "stream"})

    india_path = os.path.join(DATA_DIR, "india_specific_careers.csv")
    if os.path.exists(india_path):
        india_df = pd.read_csv(india_path)
        if "12th_stream" in india_df.columns:
            india_df = india_df.rename(columns={"12th_stream": "stream"})
        shared_cols = [c for c in df.columns if c in india_df.columns]
        india_df = india_df[shared_cols]
        df = pd.concat([df, india_df], ignore_index=True).drop_duplicates(
            subset="job_title", keep="first"
        )
        print(f"Career list after merge: {len(df)} careers")

    return df, fresh_fetch


@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_pinecone_index():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc.Index("career-discovery")


@st.cache_data
def load_students():
    return pd.read_csv(os.path.join(DATA_DIR, "students.csv"))


@st.cache_data
def load_interactions():
    return pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
