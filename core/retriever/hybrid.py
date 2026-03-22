# =========================================
# Hybrid 检索（向量 + BM25 融合）
# =========================================

def hybrid_fusion(vector_results, bm25_results, alpha=0.7):
    """
    alpha: 向量权重（大厂一般 0.6~0.8）

    返回：
    [(doc, score)]
    """

    scores = {}

    # 向量结果
    for doc, score in vector_results:
        scores[doc] = score * alpha

    # BM25结果
    for doc, score in bm25_results:
        if doc in scores:
            scores[doc] += score * (1 - alpha)
        else:
            scores[doc] = score * (1 - alpha)

    # 排序
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked