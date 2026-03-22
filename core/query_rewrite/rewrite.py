# =========================================
# Query Rewrite（查询增强）
# =========================================

def simple_rewrite(query):
    """
    基础改写（生产可替换为 LLM）
    """

    templates = [
        "{}",
        "请详细解释：{}",
        "{} 的原理",
        "{} 的应用场景"
    ]

    return list(set([t.format(query) for t in templates]))