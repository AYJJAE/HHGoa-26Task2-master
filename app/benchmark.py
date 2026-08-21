"""
Measure end-to-end retrieval latency (embed + FAISS search) against the
50ms budget defined in app/config.py.

Usage:
    python -m app.benchmark [n_queries]
"""

import statistics
import sys
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if current_dir.name == "app" and str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from app.config import LATENCY_BUDGET_MS
from app.retriever import search, warmup

QUERIES = [
    "What is FAISS used for?",
    "How does HNSW indexing work?",
    "What is retrieval augmented generation?",
    "Which embedding model is fast on CPU?",
    "How do you reduce RAG latency?",
    "What does efSearch control?",
    "Why normalize embeddings before indexing?",
    "What are the stages of a RAG pipeline?",
]


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("Warming up (model load + first inference)...")
    warmup()

    total_ms = []
    embed_ms = []
    search_ms = []

    for i in range(n):
        query = QUERIES[i % len(QUERIES)]

        resp = search(query, top_k=5)

        total_ms.append(resp.total_ms)
        embed_ms.append(resp.embed_ms)
        search_ms.append(resp.search_ms)

    print(f"\nRan {n} queries\n")

    print(
        f"{'stage':<12}"
        f"{'avg':>8}"
        f"{'p50':>8}"
        f"{'p70':>8}"
        f"{'p100':>8}"
        "   (ms)"
    )

    for name, values in [
        ("embed", embed_ms),
        ("search", search_ms),
        ("total", total_ms),
    ]:
        print(
            f"{name:<12}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 70):>8.2f}"
            f"{percentile(values, 100):>8.2f}"
        )

    p100_total = percentile(total_ms, 100)

    print(
        f"\nLatency budget: "
        f"{LATENCY_BUDGET_MS}ms | "
        f"p100 total: {p100_total:.2f}ms"
    )

    if p100_total <= LATENCY_BUDGET_MS:
        print("PASS: within budget")
    else:
        print("FAIL: over budget -- see README 'Tuning latency' section")
        sys.exit(1)


if __name__ == "__main__":
    main()
