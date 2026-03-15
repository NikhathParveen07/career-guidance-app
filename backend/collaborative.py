# ============================================
# backend/collaborative.py
# Lightweight SVD collaborative filtering
# using only numpy — no external ML library needed
# ============================================
import numpy as np
import streamlit as st
from backend.data_loader import load_interactions


@st.cache_resource
def load_svd_model():
    """
    Train a lightweight SVD model on the student-career
    interaction dataset.

    How it works:
    1. Build a student x career rating matrix R
    2. Normalise by subtracting the global mean rating
    3. Decompose R using Singular Value Decomposition
    4. Reconstruct a predicted rating matrix R_pred
    5. Return a LightSVD object with a .predict() interface

    The LightSVD class mimics the Surprise library's
    prediction interface so the rest of the code stays clean.
    """
    interactions_df = load_interactions()

    # Build index maps for students and careers
    users    = interactions_df['student_id'].unique().tolist()
    careers  = sorted(interactions_df['career_id'].unique().tolist())
    user_idx = {u: i for i, u in enumerate(users)}
    item_idx = {c: i for i, c in enumerate(careers)}

    n_users = len(users)
    n_items = len(careers)

    # Build rating matrix — 0 means no interaction
    R = np.zeros((n_users, n_items))
    for _, row in interactions_df.iterrows():
        u = user_idx.get(row['student_id'])
        i = item_idx.get(row['career_id'])
        if u is not None and i is not None:
            R[u][i] = row['rating']

    # Normalise — subtract global mean from observed ratings only
    mean_rating = R[R > 0].mean()
    R_norm      = np.where(R > 0, R - mean_rating, 0)

    # SVD decomposition — keep top 20 latent factors
    U, sigma, Vt = np.linalg.svd(R_norm, full_matrices=False)
    k      = min(20, len(sigma))
    R_pred = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :] + mean_rating

    return LightSVD(R_pred, user_idx, item_idx, mean_rating)


class LightSVD:
    """
    Lightweight SVD predictor.
    Predicts how much a student would rate a given career
    based on patterns learned from peer interactions.

    For unknown students (cold-start), returns the global
    mean rating as a neutral fallback.
    """

    def __init__(self, R_pred, user_idx, item_idx, mean_rating):
        self.R_pred      = R_pred
        self.user_idx    = user_idx
        self.item_idx    = item_idx
        self.mean_rating = mean_rating

    def predict(self, user_id, career_id):
        """
        Returns a Prediction object with .est attribute
        containing the predicted rating (1–5 scale).
        """
        u = self.user_idx.get(user_id)
        i = self.item_idx.get(career_id)

        if u is None or i is None:
            # Cold-start: student not in training data
            return _Prediction(self.mean_rating)

        predicted = float(np.clip(self.R_pred[u][i], 1, 5))
        return _Prediction(predicted)

    def get_top_careers(self, user_id, n=10):
        """
        Return the top N career indices predicted for a student.
        Used for collaborative-only recommendation mode.
        """
        u = self.user_idx.get(user_id)
        if u is None:
            return []

        scores = [
            (cid, float(self.R_pred[u][i]))
            for cid, i in self.item_idx.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


class _Prediction:
    """Simple container matching Surprise library's Prediction interface."""
    def __init__(self, est):
        self.est = est
