import numpy as np
from sentence_transformers import SentenceTransformer
import re


class SemanticChunker:
    def __init__(
        self,
        model_name="BAAI/bge-small-zh",
        similarity_threshold=0.75,
        max_chunk_size=500,
        min_chunk_size=100
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split_sentences(self, text):
        sentences = re.split(r'[。！？\n]', text)
        return [s.strip() for s in sentences if s.strip()]

    def compute_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    def chunk(self, text):
        sentences = self.split_sentences(text)

        if not sentences:
            return []

        embeddings = self.model.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]
        start = 0  # 当前chunk起点

        for i in range(1, len(sentences)):

            current_text = "".join(current_chunk)

            # ✅ 用整个chunk语义
            chunk_vec = np.mean(embeddings[start:i], axis=0)

            sim = self.compute_similarity(
                chunk_vec,
                embeddings[i]
            )

            # ✅ 切分条件
            if (
                (sim < self.similarity_threshold and len(current_text) > self.min_chunk_size)
                or len(current_text) > self.max_chunk_size
            ):
                chunks.append(current_text)

                current_chunk = [sentences[i]]
                start = i
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks
