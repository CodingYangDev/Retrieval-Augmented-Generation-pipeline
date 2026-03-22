# =========================================
# Query Pipeline（完整查询流程）
# =========================================

from core.query_rewrite.rewrite import simple_rewrite
from core.retriever.hybrid import hybrid_fusion
from core.rerank.reranker import reranker


class QueryPipeline:

    def __init__(self, vector_retriever, bm25_retriever, redis_store):
        """
        注入依赖
        """
        self.vector = vector_retriever
        self.bm25 = bm25_retriever
        self.redis = redis_store

    def run(self, query):
        """
        完整流程：

        1. cache
        2. query rewrite
        3. 多路召回
        4. hybrid融合
        5. rerank
        """

        # -------------------------
        # 1️⃣ cache
        # -------------------------
        cache_key = f"rag:{query}"
        cached = self.redis.get(cache_key)

        if cached:
            return eval(cached)

        # -------------------------
        # 2️⃣ query rewrite
        # -------------------------
        queries = simple_rewrite(query)

        vector_results = []
        bm25_results = []

        # -------------------------
        # 3️⃣ 多路召回
        # -------------------------
        for q in queries:
            vector_results.extend(self.vector.search(q))
            bm25_results.extend(self.bm25.search(q))

        # -------------------------
        # 4️⃣ hybrid
        # -------------------------
        hybrid_results = hybrid_fusion(
            vector_results,
            bm25_results
        )

        # -------------------------
        # 5️⃣ rerank
        # -------------------------
        docs = [doc for doc, _ in hybrid_results[:20]]

        reranked = reranker.rerank(query, docs)

        final_docs = reranked[:5]

        # -------------------------
        # 6️⃣ cache
        # -------------------------
        self.redis.set(cache_key, str(final_docs))

        return final_docs