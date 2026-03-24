# ============================================
# backend/collaborative.py
# LightSVD — pure numpy collaborative filtering
# Replaces scikit-surprise completely
# ============================================
import numpy as np
import streamlit as st
import random


class LightSVD:
    """
    Lightweight SVD collaborative filtering using numpy only.
    Drop-in replacement for scikit-surprise SVD.
    Compatible with Kaggle numpy 2.x and all environments.
    """
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors   = n_factors
        self.n_epochs    = n_epochs
        self.lr          = lr
        self.reg         = reg
        self.global_mean = 3.5

    def fit(self, ratings_df):
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
                err        = row['rating'] - self._raw(u, i)
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu         = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * pu        - self.reg * self.Q[i])

        return self

    def _raw(self, u_idx, i_idx):
        return (self.global_mean +
                self.bu[u_idx] +
                self.bi[i_idx] +
                np.dot(self.P[u_idx], self.Q[i_idx]))

    def predict(self, user_id, career_id):
        """
        Predict rating for user_id and career_id.
        Returns float between 1 and 5.
        Compatible with scikit-surprise predict() interface.
        """
        u = self.user_index.get(user_id)
        i = self.item_index.get(career_id)
        if u is None or i is None:
            return self.global_mean
        return float(np.clip(self._raw(u, i), 1, 5))


def _generate_interactions(df):
    """
    Generate synthetic student-career interactions
    for training LightSVD on 1016 O*NET careers.
    """
    import pandas as pd
    random.seed(42)
    np.random.seed(42)

    students     = [f"STU{i:04d}" for i in range(200)]
    interactions = []

    for student_id in students:
        stream         = random.choice(['Science', 'Commerce', 'Arts', 'Vocational'])
        stream_careers = df[df['stream'] == stream].index.tolist()
        other_careers  = df[df['stream'] != stream].index.tolist()

        for career_id in random.sample(stream_careers, min(15, len(stream_careers))):
            interactions.append({
                'student_id': student_id,
                'career_id':  career_id,
                'rating':     random.choice([4, 4, 5, 5, 3])
            })
        for career_id in random.sample(other_careers, min(5, len(other_careers))):
            interactions.append({
                'student_id': student_id,
                'career_id':  career_id,
                'rating':     random.choice([1, 2, 2, 3])
            })

    return pd.DataFrame(interactions)


@st.cache_resource
def load_svd_model():
    """
    Load and return trained LightSVD model.
    Cached by Streamlit — trains once per session.
    """
    from backend.data_loader import load_careers

    df = load_careers()

    # Ensure stream column exists
    if '12th_stream' in df.columns and 'stream' not in df.columns:
        df = df.rename(columns={'12th_stream': 'stream'})

    interactions_df = _generate_interactions(df)

    model = LightSVD(n_factors=50, n_epochs=20)
    model.fit(interactions_df)

    return model
