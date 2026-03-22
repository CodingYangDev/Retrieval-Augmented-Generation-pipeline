# =========================================
# LLM 封装（支持 OpenAI / 本地模型）
# =========================================

import requests


class LLM:
    def __init__(self, api_url="http://localhost:11434/api/generate"):
        """
        默认使用 Ollama（本地模型）
        你也可以换成 OpenAI
        """
        self.api_url = api_url

    def generate(self, prompt):
        """
        调用 LLM 生成
        """

        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(self.api_url, json=payload)

        if response.status_code != 200:
            return "LLM 调用失败"

        return response.json().get("response", "")


# 单例
llm = LLM()
