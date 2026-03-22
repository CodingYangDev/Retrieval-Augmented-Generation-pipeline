# =========================================
# Ingest Pipeline（企业级版）
# =========================================

from core.chunking.parent_child import build_hierarchical_chunks
from core.embedding.embedder import embedder


class IngestPipeline:

    def __init__(self, milvus_store, mongo_store):
        self.milvus = milvus_store
        self.mongo = mongo_store

    def run(self, text):
        """
        企业级流程：

        1. Parent 切片
        2. Child 切片
        3. 存 Parent（Mongo）
        4. Child 向量化
        5. 存 Milvus
        """

        # -------------------------
        # 1️⃣ 构建父子块
        # -------------------------
        parents, children = build_hierarchical_chunks(text)

        # -------------------------
        # 2️⃣ 存 Parent（Mongo）
        # -------------------------
        for p in parents:
            self.mongo.save_parent(
                p["parent_id"],
                p["text"]
            )

        # -------------------------
        # 3️⃣ 准备向量数据
        # -------------------------
        texts = [c["chunk"] for c in children]
        parent_ids = [c["parent_id"] for c in children]

        # -------------------------
        # 4️⃣ 向量化
        # -------------------------
        embeddings = embedder.embed(texts)

        # -------------------------
        # 5️⃣ 写入 Milvus
        # -------------------------
        self.milvus.insert(
            embeddings,
            texts,
            parent_ids
        )

        return {
            "parent_count": len(parents),
            "child_count": len(children)
        }