# =========================================
# RAG Pipeline（最终闭环）
# =========================================

from core.pipeline.query_pipeline import QueryPipeline
from core.prompt.prompt_template import build_prompt
from core.llm.llm import llm


class RagPipeline:

    def __init__(self, query_pipeline):
        """
        注入 QueryPipeline
        """
        self.query_pipeline = query_pipeline

    def run(self, query):
        """
        完整流程：

        1. 检索
        2. Prompt构建
        3. LLM生成
        """

        # 1️⃣ 检索
        docs = self.query_pipeline.run(query)

        # 2️⃣ Prompt
        prompt = build_prompt(query, docs)

        # 3️⃣ LLM生成
        answer = llm.generate(prompt)

        return {
            "answer": answer,
            "docs": docs
        }