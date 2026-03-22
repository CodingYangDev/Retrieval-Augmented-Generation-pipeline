# =========================================
# 全局配置文件（所有模块依赖这里）
# 配置：统一管理配置，避免硬编码
# =========================================

class Settings:
    # -------------------------------
    # Milvus 向量数据库配置
    # -------------------------------
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
    COLLECTION_NAME = "rag_collection"

    # -------------------------------
    # MongoDB（存父文档）
    # -------------------------------
    MONGO_URI = "mongodb://localhost:27017"
    DB_NAME = "rag_db"

    # -------------------------------
    # Redis（缓存）
    # -------------------------------
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379

    # -------------------------------
    # 向量模型配置
    # -------------------------------
    EMBEDDING_MODEL = "BAAI/bge-small-zh"
    RERANK_MODEL = "BAAI/bge-reranker-base"


# 单例（全局使用）
settings = Settings()