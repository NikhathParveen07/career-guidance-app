# ============================================
# backend/data_loader.py
# Loads career data from O*NET API via onet_loader
# Falls back to CSV if O*NET unavailable
# Also loads sentence model, Pinecone, Supabase
# ============================================
import os
import sys
import pandas as pd
import streamlit as st
from pinecone import Pinecone
from supabase import create_client
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onet_loader import fetch_all_onet_careers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


@st.cache_resource
def load_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )


@st.cache_data(ttl=86400)  # refresh every 24 hours
def load_careers():
    """
    Load career data from O*NET API via Supabase cache.
    Falls back to local CSV if O*NET credentials unavailable.

    Returns pandas DataFrame with columns:
        onet_code, job_title, sector, 12th_stream,
        primary_riasec, secondary_riasec, core_skills
    """
    # Try O*NET via Supabase
    try:
        username = st.secrets.get("ONET_USERNAME", None)
        password = st.secrets.get("ONET_PASSWORD", None)
        supabase = load_supabase()

        if username and password:
            careers = fetch_all_onet_careers(username, password, supabase)
            if careers and len(careers) > 10:
                df = pd.DataFrame(careers)
                # Rename for compatibility with rest of system
                if "12th_stream" in df.columns:
                    df = df.rename(columns={"12th_stream": "stream"})
                if "onet_code" in df.columns and "nco_id" not in df.columns:
                    df["nco_id"] = df["onet_code"]
                return df
    except Exception as e:
        st.warning(f"O*NET unavailable, loading from CSV: {e}")

    # Fallback to CSV
    csv_path = os.path.join(DATA_DIR, "career_master_final.csv")
    df = pd.read_csv(csv_path)
    if "12th_stream" in df.columns:
        df = df.rename(columns={"12th_stream": "stream"})
    return df


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
