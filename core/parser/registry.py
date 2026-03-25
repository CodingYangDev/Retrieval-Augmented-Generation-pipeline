

class ParserRegistry:

    def __init__(self):
        self._parsers = {}

    def register(self, ext: str, parser):
        self._parsers[ext.lower()] = parser

    def get_parser(self, ext: str):
        parser = self._parsers.get(ext.lower())
        if not parser:
            raise ValueError(f"Unsupported file type: {ext}")
        return parser