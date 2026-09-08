"""Measure recommendation latency."""
import time
import sys

sys.path.insert(0, ".")

from ml_engine.content_similarity import ContentSimilarityEngine
import pandas as pd

# Load data
df = pd.read_csv("../data/movies_dataset.csv").fillna("")
for col in ["genres", "cast", "director", "keywords", "overview"]:
    if col not in df.columns:
        df[col] = ""
df["source"] = "local"
for col in ["genres", "cast", "director", "keywords"]:
    df[col] = df[col].astype(str).str.replace(r"[\[\]'\"]", "", regex=True)

# Measure engine build time
t0 = time.time()
engine = ContentSimilarityEngine(df)
build_time = time.time() - t0
print(f"Engine build time: {build_time:.2f}s")

# Measure single recommendation latency (10 runs, averaged)
movie_ids = df["id"].astype(int).tolist()[:50]
times = []
for mid in movie_ids[:10]:
    t0 = time.time()
    engine.get_similar(mid, top_n=10)
    times.append(time.time() - t0)

avg_latency = sum(times) / len(times)
print(f"Avg single recommendation latency (top-10): {avg_latency*1000:.1f}ms")

# Measure batch similarity (the way benchmark does it)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

idxs = list(range(64))
t0 = time.time()
meta_scores = cosine_similarity(engine.meta_matrix[idxs], engine.meta_matrix)
overview_scores = cosine_similarity(engine.overview_matrix[idxs], engine.overview_matrix)
combined = 0.72 * meta_scores + 0.28 * overview_scores
batch_time = time.time() - t0
print(f"Batch similarity (64 seeds x {len(df)} movies): {batch_time:.2f}s")

# Memory estimate
meta_nnz = engine.meta_matrix.nnz
overview_nnz = engine.overview_matrix.nnz
meta_shape = engine.meta_matrix.shape
overview_shape = engine.overview_matrix.shape
print(f"Meta matrix: {meta_shape}, nnz={meta_nnz}")
print(f"Overview matrix: {overview_shape}, nnz={overview_nnz}")
print(f"Meta vocab size: {len(engine.meta_vectorizer.vocabulary_)}")
print(f"Overview vocab size: {len(engine.overview_vectorizer.vocabulary_)}")
