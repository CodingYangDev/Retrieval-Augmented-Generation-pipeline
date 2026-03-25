from .base import BaseParser


class MarkdownParser(BaseParser):

    def parse(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()