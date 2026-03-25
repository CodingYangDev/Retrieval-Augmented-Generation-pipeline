import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from .base import BaseParser


class PDFParser(BaseParser):

    def parse(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        texts = []

        for page in doc:
            text = page.get_text()

            if not text.strip():  # 如果是扫描版PDF，使用OCR
                text = self._ocr_page(page)

            texts.append(text)

        return "\n".join(texts)

    def _ocr_page(self, page):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)