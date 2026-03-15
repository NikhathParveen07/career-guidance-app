# ============================================
# backend/data_loader.py
# Load CSVs and connect to Pinecone + Supabase
# ============================================
import pandas as pd
import streamlit as st
from pinecone import Pinecone
from supabase import create_client
from sentence_transformers import SentenceTransformer


@st.cache_data
def load_careers():
    """Load the 90-career knowledge base from CSV."""
    return pd.read_csv("data/career_master_final.csv")


@st.cache_data
def load_students():
    """Load synthetic student profiles."""
    return pd.read_csv("data/students.csv")


@st.cache_data
def load_interactions():
    """Load student-career interaction ratings."""
    return pd.read_csv("data/interactions.csv")


@st.cache_resource
def load_sentence_model():
    """Load the sentence transformer model for embedding generation."""
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_pinecone_index():
    """Connect to Pinecone and return the career-discovery index."""
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc.Index("career-discovery")


@st.cache_resource
def load_supabase():
    """Connect to Supabase and return the client."""
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
