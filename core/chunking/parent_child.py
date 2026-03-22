# =========================================
# 父子块结构（企业级核心设计）
# 小块用于检索，大块用于生成
# =========================================

# =========================================
# Parent + Child 分块
# =========================================

import uuid
from core.chunking.sliding_window import sliding_window_chunk


def build_hierarchical_chunks(
    text,
    parent_chunk_size=1200,
    parent_overlap=200,
    child_chunk_size=300,
    child_overlap=50
):
    """
    构建企业级父子块结构

    返回：
    parents: List[dict]
    children: List[dict]
    """

    parents = []
    children = []

    # -------------------------
    # 1️⃣ 先切 Parent（大块）
    # -------------------------
    parent_chunks = sliding_window_chunk(
        text,
        chunk_size=parent_chunk_size,
        overlap=parent_overlap
    )

    # -------------------------
    # 2️⃣ 每个 Parent 再切 Child
    # -------------------------
    for parent_text in parent_chunks:
        parent_id = str(uuid.uuid4())

        # 存 parent
        parents.append({
            "parent_id": parent_id,
            "text": parent_text
        })

        # 切 child
        child_chunks = sliding_window_chunk(
            parent_text,
            chunk_size=child_chunk_size,
            overlap=child_overlap
        )

        for child_text in child_chunks:
            children.append({
                "chunk": child_text,
                "parent_id": parent_id
            })

    return parents, children