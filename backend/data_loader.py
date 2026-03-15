# ============================================
# backend/data_loader.py
# ============================================
import os
import pandas as pd
import streamlit as st
from pinecone import Pinecone
from supabase import create_client
from sentence_transformers import SentenceTransformer

# Absolute path to data folder — works on any machine
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")


@st.cache_data
def load_careers():
    return pd.read_csv(os.path.join(DATA_DIR, "career_master_final.csv"))


@st.cache_data
def load_students():
    return pd.read_csv(os.path.join(DATA_DIR, "students.csv"))


@st.cache_data
def load_interactions():
    return pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))


@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_pinecone_index():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc.Index("career-discovery")


@st.cache_resource
def load_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
