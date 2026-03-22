# =========================================
# BM25 检索（支持中文分词）
# =========================================

from rank_bm25 import BM25Okapi
import jieba


class BM25Retriever:
    def __init__(self):
        """
        初始化 BM25
        """
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

    def add_documents(self, docs):
        """
        动态加入语料（生产中一般用 ES）
        """

        self.documents.extend(docs)

        # 中文分词
        self.tokenized_corpus = [
            list(jieba.cut(doc)) for doc in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, topk=10):
        """
        BM25 检索
        """

        if not self.bm25:
            return []

        tokenized_query = list(jieba.cut(query))

        scores = self.bm25.get_scores(tokenized_query)

        results = list(zip(self.documents, scores))

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:topk]