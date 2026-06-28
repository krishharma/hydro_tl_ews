"""Hydrological-similarity clustering for source-domain selection.

Implements the Ougahi & Rowan (2026) recommendation: pre-train on basins
that are *most similar* to the target rather than indiscriminately on all
available donors.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_basins(attributes: pd.DataFrame, n_clusters: int = 8,
                   random_state: int = 42) -> pd.Series:
    """Run k-means on standardized catchment attributes."""
    X = StandardScaler().fit_transform(attributes.fillna(attributes.median()))
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)
    return pd.Series(labels, index=attributes.index, name="cluster")


def select_donor_basins(attributes: pd.DataFrame, target_basin: str,
                        n_donors: int = 50) -> list[str]:
    """Return the ``n_donors`` basins closest to ``target_basin`` in attribute space."""
    if target_basin not in attributes.index:
        raise KeyError(f"Target basin {target_basin} missing from attribute table.")
    X = attributes.fillna(attributes.median())
    Z = StandardScaler().fit_transform(X)
    Zdf = pd.DataFrame(Z, index=X.index, columns=X.columns)
    target_vec = Zdf.loc[target_basin].to_numpy()
    dists = np.linalg.norm(Zdf.to_numpy() - target_vec, axis=1)
    order = np.argsort(dists)
    donors = [bid for bid in Zdf.index[order] if bid != target_basin][:n_donors]
    return donors
