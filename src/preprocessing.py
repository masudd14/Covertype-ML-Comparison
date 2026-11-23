"""Preprocessing transformations: standard scaling and PCA dimensionality reduction."""

from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd


def load_transformers(models_dir: str) -> Tuple[object, object]:
    """Load pre-fitted StandardScaler and PCA model.

    Args:
        models_dir: Directory containing scaler.pkl and pca_model.pkl

    Returns:
        Tuple of (scaler, pca_model)
    """
    path = Path(models_dir)
    scaler_path = path / "scaler.pkl"
    pca_path = path / "pca_model.pkl"

    if not scaler_path.exists() or not pca_path.exists():
        raise FileNotFoundError("scaler.pkl or pca_model.pkl missing from models directory.")

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    return scaler, pca


def transform_features(features: np.ndarray, scaler: object, pca: object) -> np.ndarray:
    """Apply standard scaling followed by PCA transformation.

    Args:
        features: Input feature matrix (N x 54).
        scaler: Fitted StandardScaler instance.
        pca: Fitted PCA instance.

    Returns:
        Dimensionality-reduced feature matrix.
    """
    scaled = scaler.transform(features)
    reduced = pca.transform(scaled)
    return reduced
