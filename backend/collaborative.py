# ============================================
# backend/collaborative.py
# LightSVD — pure numpy collaborative filtering
# ============================================
import numpy as np
import pandas as pd
import streamlit as st
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

REAL_INTERACTION_WEIGHT = 2


class LightSVD:
    """
    Lightweight SVD collaborative filtering using numpy only.
    """
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors   = n_factors
        self.n_epochs    = n_epochs
        self.lr          = lr
        self.reg         = reg
        self.global_mean = 3.5

    def fit(self, ratings_df):
        if ratings_df.empty:
            # Nothing to train on — keep defaults so predict() returns global_mean
            self.user_index = {}
            self.item_index = {}
            self.P  = np.array([])
            self.Q  = np.array([])
            self.bu = np.array([])
            self.bi = np.array([])
            return self

        users  = ratings_df['student_id'].unique()
        items  = ratings_df['career_id'].unique()

        self.user_index  = {u: i for i, u in enumerate(users)}
        self.item_index  = {it: i for i, it in enumerate(items)}
        self.global_mean = ratings_df['rating'].mean()

        n_users = len(users)
        n_items = len(items)

        np.random.seed(42)
        self.P  = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q  = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)

        for epoch in range(self.n_epochs):
            for _, row in ratings_df.iterrows():
                u = self.user_index.get(row['student_id'])
                i = self.item_index.get(row['career_id'])
                if u is None or i is None:
                    continue
                weight     = row.get('weight', 1.0)
                err        = row['rating'] - self._raw(u, i)
                self.bu[u] += self.lr * (weight * err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (weight * err - self.reg * self.bi[i])
                pu         = self.P[u].copy()
                self.P[u] += self.lr * (weight * err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (weight * err * pu        - self.reg * self.Q[i])

        return self

    def _raw(self, u_idx, i_idx):
        return (self.global_mean +
                self.bu[u_idx] +
                self.bi[i_idx] +
                np.dot(self.P[u_idx], self.Q[i_idx]))

    def predict(self, user_id, career_id):
        u = self.user_index.get(user_id)
        i = self.item_index.get(career_id)
        if u is None or i is None:
            return self.global_mean
        # Guard against uninitialized arrays (empty training set)
        if self.P.size == 0 or self.Q.size == 0:
            return self.global_mean
        return float(np.clip(self._raw(u, i), 1, 5))


def _load_seed_interactions():
    try:
        csv_path = os.path.join(DATA_DIR, "interactions.csv")
        df = pd.read_csv(csv_path)
        df = df[['student_id', 'career_id', 'rating']].copy()
        df['weight'] = 1.0
        df['source'] = 'seed'
        return df
    except Exception as e:
        print(f"Could not load seed interactions: {e}")
        return pd.DataFrame(columns=['student_id', 'career_id', 'rating', 'weight', 'source'])


def _load_real_interactions(supabase):
    try:
        result = supabase.table("live_interactions").select("*").execute()
        if not result.data:
            return pd.DataFrame(columns=['student_id', 'career_id', 'rating', 'weight', 'source'])

        df = pd.DataFrame(result.data)

        # Validate required columns exist before subsetting
        required = {'student_id', 'career_id', 'rating'}
        if not required.issubset(set(df.columns)):
            print(f"live_interactions table missing columns. Found: {df.columns.tolist()}")
            return pd.DataFrame(columns=['student_id', 'career_id', 'rating', 'weight', 'source'])

        df = df[['student_id', 'career_id', 'rating']].copy()
        df['weight'] = float(REAL_INTERACTION_WEIGHT)
        df['source'] = 'real'
        return df
    except Exception as e:
        print(f"Could not load real interactions from Supabase: {e}")
        return pd.DataFrame(columns=['student_id', 'career_id', 'rating', 'weight', 'source'])


def _build_combined_matrix(supabase):
    seed_df = _load_seed_interactions()
    real_df = _load_real_interactions(supabase)

    n_real = len(real_df)
    n_seed = len(seed_df)
    print(f"Interactions — seed: {n_seed}, real: {n_real}")

    if n_real == 0:
        return seed_df

    if n_seed > 0:
        real_keys = set(zip(real_df['student_id'], real_df['career_id']))
        seed_df = seed_df[
            ~seed_df.apply(
                lambda r: (r['student_id'], r['career_id']) in real_keys,
                axis=1
            )
        ]

    combined = pd.concat([seed_df, real_df], ignore_index=True)
    return combined


def save_interaction(supabase, student_id, career_id, career_title, signal, stream):
    """
    Save a Keep (5) or Drop (1) interaction to Supabase.
    Returns True on success, False on failure.
    """
    try:
        supabase.table("live_interactions").upsert({
            "student_id":   student_id,
            "career_id":    int(career_id),
            "career_title": career_title,
            "rating":       int(signal),
            "stream":       stream,
        }, on_conflict="student_id,career_id").execute()
        return True
    except Exception as e:
        print(f"Could not save interaction: {e}")
        return False


@st.cache_resource
def load_svd_model(_supabase):
    """
    Load and return trained LightSVD model.
    _supabase prefixed with underscore so Streamlit skips hashing it.
    """
    combined_df = _build_combined_matrix(_supabase)
    model = LightSVD(n_factors=50, n_epochs=20)
    model.fit(combined_df)
    return model
