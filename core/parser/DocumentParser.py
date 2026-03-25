# core/parser/document_parser.py

import os
import tempfile

from core.parser.excel_parser import ExcelParser
from core.parser.markdown_parser import MarkdownParser
from core.parser.pdf_parser import PDFParser
from core.parser.ppt_parser import PPTParser
from core.parser.word_parser import WordParser
from core.parser.registry import ParserRegistry


class DocumentParser:

    def __init__(self):
        self.registry = ParserRegistry()

        # ✅ 注册解析器（企业标准）
        self.registry.register(".pdf", PDFParser())
        self.registry.register(".docx", WordParser())
        self.registry.register(".pptx", PPTParser())
        self.registry.register(".md", MarkdownParser())
        self.registry.register(".txt", MarkdownParser())  # txt也可以复用
        self.registry.register(".xlsx", ExcelParser())

    def _save_temp_file(self, upload_file):
        """🔥 把 UploadFile 存成临时文件（关键步骤）"""
        suffix = os.path.splitext(upload_file.filename)[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload_file.file.read())
            return tmp.name

    def parse(self, upload_file):
        """
        输入：UploadFile
        输出：统一文本
        """

        # ✅ 1. 保存临时文件
        file_path = self._save_temp_file(upload_file)

        # ✅ 2. 获取解析器
        ext = os.path.splitext(file_path)[-1].lower()
        parser = self.registry.get_parser(ext)

        # ✅ 3. 执行解析
        text = parser.parse(file_path)

        return {
            "text": text,
            "source": upload_file.filename
        }