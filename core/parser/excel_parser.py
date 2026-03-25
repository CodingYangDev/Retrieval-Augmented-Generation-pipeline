import pandas as pd
from .base import BaseParser


class ExcelParser(BaseParser):

    def parse(self, file_path: str) -> str:
        df = pd.read_excel(file_path)
        return df.to_string(index=False)