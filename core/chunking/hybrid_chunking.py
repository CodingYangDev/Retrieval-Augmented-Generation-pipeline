import uuid
from typing import List, Tuple
from semantic_chunker import SemanticChunker
from sliding_window import sliding_window_chunk


class HybridChunker:

    def __init__(
        self,
        semantic_threshold=0.75,
        parent_chunk_size=1200,
        parent_overlap=200,
        child_chunk_size=300,
        child_overlap=50
    ):
        self.semantic_chunker = SemanticChunker(
            similarity_threshold=semantic_threshold
        )

        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap

    def build(self, text: str) -> Tuple[List[dict], List[dict]]:
        parents = []
        children = []

        # Step 1：语义切片
        semantic_chunks = self.semantic_chunker.chunk(text)

        # Step 2：构建 Parent + Child
        for chunk in semantic_chunks:

            parent_chunks = sliding_window_chunk(
                chunk,
                chunk_size=self.parent_chunk_size,
                overlap=self.parent_overlap
            )

            for parent_text in parent_chunks:
                parent_id = str(uuid.uuid4())

                parents.append({
                    "parent_id": parent_id,
                    "text": parent_text
                })

                child_chunks = sliding_window_chunk(
                    parent_text,
                    chunk_size=self.child_chunk_size,
                    overlap=self.child_overlap
                )

                for child_text in child_chunks:
                    children.append({
                        "chunk": child_text,
                        "parent_id": parent_id
                    })

        return parents, children