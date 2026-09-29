"""Fits the model and hands back a probability, not a verdict.

The verdict is somebody else's job. This file is not allowed to decide
anything, which is why it returns a number between 0 and 1 and stops.
"""

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cat_check import features


def fit(pictures, is_target, top=255.0, components=30, seed=1):
    rows = features.build(pictures, top)
    scaler = StandardScaler().fit(rows)
    squeeze = PCA(
        n_components=min(components, rows.shape[1], len(rows)), random_state=seed
    ).fit(scaler.transform(rows))
    engine = LogisticRegression(max_iter=2000, random_state=seed)
    engine.fit(squeeze.transform(scaler.transform(rows)), is_target)
    return {"scaler": scaler, "squeeze": squeeze, "engine": engine, "top": top}


def chance_of_target(fitted, pictures):
    return fitted["engine"].predict_proba(small(fitted, pictures))[:, 1]


def small(fitted, pictures):
    """The picture in the few numbers the model actually looks at."""
    rows = features.build(pictures, fitted["top"])
    return fitted["squeeze"].transform(fitted["scaler"].transform(rows))
