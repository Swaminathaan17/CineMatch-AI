"""Baseline benchmark runner with timing."""
import time
import json
import sys

sys.path.insert(0, ".")

from evaluation.benchmark import run_benchmark

start = time.time()
result = run_benchmark(seed_count=500, ks=[5, 10, 20])
elapsed = time.time() - start

print("=== BASELINE BENCHMARK ===")
print(f"Benchmark runtime: {elapsed:.1f}s")
print(f"Dataset size: {result['benchmark']['dataset_size']}")
print(f"Seeds: {result['benchmark']['seed_count']}")
for k in ["5", "10", "20"]:
    m = result["metrics"][k]
    print(f"--- @{k} ---")
    for metric in ["precision", "ndcg", "map", "mrr"]:
        v = m[metric]
        print(f"  {metric}@{k}: {v['current']:.4f} (CI: [{v['current_ci95'][0]:.4f}, {v['current_ci95'][1]:.4f}])")
print(f"Coverage@20: {result['diagnostics']['catalog_coverage_at_max_k']:.4f}")

with open("../reports/baseline_500.json", "w") as f:
    json.dump(result, f, indent=2)
print("Saved to reports/baseline_500.json")
