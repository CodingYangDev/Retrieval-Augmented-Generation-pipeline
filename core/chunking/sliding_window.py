# =========================================
# 滑动窗口切片
# 适用于长文本（论文 / 法律 / 大文档）
# =========================================

def sliding_window_chunk(text, chunk_size=300, overlap=50):
    """
    参数：
    text: 原始文本
    chunk_size: 每块长度
    overlap: 重叠长度（防止语义断裂）

    返回：
    List[str]
    """

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        chunk = text[start:end]
        chunks.append(chunk)

        # 滑动窗口核心
        start += chunk_size - overlap

    return chunks