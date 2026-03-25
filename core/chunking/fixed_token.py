import tiktoken
from typing import List


class TokenChunker:

    def __init__(
        self,
        model_name="gpt-4o-mini",
        chunk_size=300,
        overlap=50
    ):
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            start += self.chunk_size - self.overlap

        return chunks
