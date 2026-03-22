# =========================================
# 段落切片
# 适用于：Markdown / 技术文档
# =========================================

def paragraph_chunk(text):
    """
    按段落切分（两个换行）

    返回：
    List[str]
    """
    paragraphs = text.split("\n\n")

    # 去掉空内容
    return [p.strip() for p in paragraphs if p.strip()]