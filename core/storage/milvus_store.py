# =========================================
# Milvus 向量数据库操作
# 包含：连接 / 建表 / 建索引 / 插入 / 查询
# =========================================

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from config.settings import settings

# embedding 维度（必须和模型一致）
DIM = 512


class MilvusStore:

    def __init__(self):
        """
        初始化 Milvus 连接 + Collection
        """
        self._connect()
        self.collection = self._init_collection()

    def _connect(self):
        """
        连接 Milvus
        """
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )

    def _init_collection(self):
        """
        初始化 Collection（如果不存在就创建）
        """

        # 如果存在就直接用
        if utility.has_collection(settings.COLLECTION_NAME):
            collection = Collection(settings.COLLECTION_NAME)
            collection.load()
            return collection

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

            # 向量字段（用于检索）
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),

            # 存 chunk 文本（方便 debug / rerank）
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),

            # 父文档ID（用于回溯大文本）
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
        ]

        schema = CollectionSchema(fields, description="RAG Collection")

        collection = Collection(
            name=settings.COLLECTION_NAME,
            schema=schema
        )

        # 创建索引（IVF_FLAT：通用稳定）
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 128}
        }

        collection.create_index("embedding", index_params)

        # 加载到内存（必须）
        collection.load()

        return collection

    # -------------------------------------
    # 插入数据
    # -------------------------------------
    def insert(self, embeddings, texts, parent_ids):
        """
        插入数据（顺序必须一致）

        embeddings: List[List[float]]
        texts: List[str]
        parent_ids: List[str]
        """

        self.collection.insert([
            embeddings.tolist(),  # ⚠️ 必须转list
            texts,
            parent_ids
        ])

    # -------------------------------------
    # 向量检索
    # -------------------------------------
    def search(self, query_vector, topk=10):
        """
        向量搜索
        """

        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={
                "metric_type": "IP",
                "params": {"nprobe": 10}
            },
            limit=topk,
            output_fields=["text", "parent_id"]
        )

        docs = []

        for hits in results:
            for hit in hits:
                docs.append((
                    hit.entity.get("text"),
                    hit.score
                ))

        return docs