# RAG Benchmark Results

## Retrieval Quality

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- | --- |
| Hybrid RRF (Passage) | 33.0% | 68.0% | 80.0% | 0.4855 |

## Latency (P50 / P70 / P100 in ms)

| Configuration | Total RAG | Embedding | Dense | Sparse | RRF | Reranking | Context |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid RRF (Passage) | 50.42 / 54.94 / 65.44 | 14.19 / 17.05 / 25.72 | 5.86 / 6.19 / 9.17 | 29.17 / 31.56 / 38.26 | 0.24 / 0.27 / 2.71 | - / - / - | 0.01 / 0.01 / 0.01 |
