"""Step B: Hybrid Retrieval.

BM25 (rank_bm25, top-100) + dense retrieval (FAISS flat L2, top-100) fused
via RRF (k=60) → top-50 candidates. Dispatches on cfg.step_b.retrieval_mode:
  rrf | bm25_only | dense_only
"""
# TODO: implement
