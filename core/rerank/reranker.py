# =========================================
# Rerank（交叉编码器）
# 提升 TopK 精度
# =========================================

from sentence_transformers import CrossEncoder
from config.settings import settings


class Reranker:
    def __init__(self):
        """
        初始化 rerank 模型
        """
        self.model = CrossEncoder(settings.RERANK_MODEL)

    def rerank(self, query, docs):
        """
        对候选文档进行重新排序

        参数：
        query: str
        docs: List[str]
        """

        if not docs:
            return []

        pairs = [[query, doc] for doc in docs]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked]


# 单例
reranker = Reranker()