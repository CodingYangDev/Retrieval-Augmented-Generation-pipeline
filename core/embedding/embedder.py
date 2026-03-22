# =========================================
# 向量化模块（Embedding）
# 统一封装：统一封装模型调用
# =========================================

from sentence_transformers import SentenceTransformer
from config.settings import settings


class Embedder:
    def __init__(self):
        """
        初始化向量模型
        """
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def embed(self, texts):
        """
        批量向量化

        参数：
        texts: List[str]

        返回：
        List[vector]
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True  # 非常重要（相似度稳定）
        )


# 单例（避免重复加载模型）
embedder = Embedder()