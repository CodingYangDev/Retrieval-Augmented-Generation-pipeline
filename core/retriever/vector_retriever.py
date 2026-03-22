# =========================================
# 向量检索封装（基于 Milvus）
# =========================================

from core.embedding.embedder import embedder


class VectorRetriever:
    def __init__(self, milvus_store):
        """
        注入 Milvus 依赖
        """
        self.milvus = milvus_store

    def search(self, query, topk=10):
        """
        执行向量检索

        流程：
        1. query 向量化
        2. Milvus 检索
        """

        # 1️⃣ 向量化
        query_vec = embedder.embed([query])[0]

        # 2️⃣ 检索
        results = self.milvus.search(query_vec, topk)

        return results