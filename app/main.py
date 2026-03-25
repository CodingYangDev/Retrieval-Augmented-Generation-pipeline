# =========================================
# FastAPI 主服务（企业入口）
# =========================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import chardet
import json

from core.parser.DocumentParser import DocumentParser
# 存储层
from core.storage.milvus_store import MilvusStore
from core.storage.mongo_store import MongoStore
from core.storage.redis_store import RedisStore

# pipeline
from core.pipeline.ingest_pipeline import IngestPipeline
from core.pipeline.query_pipeline import QueryPipeline
from core.pipeline.rag_pipeline import RagPipeline

# retriever
from core.retriever.vector_retriever import VectorRetriever
from core.retriever.bm25_retriever import BM25Retriever

# 分块相关
from core.chunking.sliding_window import sliding_window_chunk
from core.chunking.parent_child import build_hierarchical_chunks
from core.embedding.embedder import embedder

app = FastAPI()

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],   # 你的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 初始化所有组件
# =========================

milvus = MilvusStore()
mongo = MongoStore()
redis_store = RedisStore()

vector_retriever = VectorRetriever(milvus)
bm25_retriever = BM25Retriever()

query_pipeline = QueryPipeline(
    vector_retriever,
    bm25_retriever,
    redis_store
)

rag_pipeline = RagPipeline(query_pipeline)
ingest_pipeline = IngestPipeline(milvus, mongo)


# =========================
# 辅助分块函数
# =========================

def fixed_size_chunk(text: str, chunk_size: int, overlap: int):
    parent_id = "fixed_parent_0"
    parent = {"parent_id": parent_id, "text": text}
    chunks = sliding_window_chunk(text, chunk_size=chunk_size, overlap=overlap)
    children = [{"chunk": chunk, "parent_id": parent_id} for chunk in chunks]
    return [parent], children

def semantic_chunk(text: str, similarity_threshold: float):
    # 简化演示：按句子分割，根据阈值合并（实际可用句向量）
    import re
    sentences = re.split(r'(?<=[。！？])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        parent_id = "semantic_parent_0"
        parent = {"parent_id": parent_id, "text": text}
        children = [{"chunk": text, "parent_id": parent_id}]
        return [parent], children
    # 简单将阈值映射为最大块大小
    max_chunk_size = max(100, int(1000 * (1 - similarity_threshold)))
    merged = []
    current = sentences[0]
    for s in sentences[1:]:
        if len(current) + len(s) <= max_chunk_size:
            current += s
        else:
            merged.append(current)
            current = s
    merged.append(current)
    parent_id = "semantic_parent_0"
    parent = {"parent_id": parent_id, "text": text}
    children = [{"chunk": chunk, "parent_id": parent_id} for chunk in merged]
    return [parent], children


# =========================
# 接口
# =========================

@app.post("/ingest")
def ingest(text: str):
    bm25_retriever.add_documents([text])
    result = ingest_pipeline.run(text)
    return result

@app.get("/query")
def query(q: str):
    result = rag_pipeline.run(q)
    return result

# 初始化解析器
parser = DocumentParser()

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    params: str = Form(...)
):
    # 1. 解析文件
    try:
        parsed = parser.parse(file)
        text = parsed["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件解析失败: {str(e)}")

    # 2. 解析分块参数
    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="参数格式错误")

    chunk_method = params_dict.get("chunk_method", "parent_child")

    # 3. 根据选择的分块方法执行分块操作
    try:
        if chunk_method == "parent_child":
            # 父子模式分块逻辑
            parents, children = build_hierarchical_chunks(
                text,
                parent_chunk_size=params_dict.get("parent_size", 1200),
                parent_overlap=params_dict.get("parent_overlap", 200),
                child_chunk_size=params_dict.get("child_size", 300),
                child_overlap=params_dict.get("child_overlap", 50)
            )
        elif chunk_method == "semantic":
            # 语义化分块逻辑
            parents, children = semantic_chunk(
                text,
                params_dict.get("similarity_threshold", 0.6)
            )
        else:
            raise HTTPException(status_code=400, detail="不支持的分块方式")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分片失败: {str(e)}")

    # 4. 存储父块到 MongoDB
    for parent in parents:
        mongo.save_parent(parent["parent_id"], parent["text"])

    # 5. 向量化子块并存储到 Milvus
    child_texts = [c["chunk"] for c in children]
    child_parent_ids = [c["parent_id"] for c in children]

    embeddings = embedder.embed(child_texts)
    milvus.insert(embeddings, child_texts, child_parent_ids)

    return {
        "filename": file.filename,
        "parent_count": len(parents),
        "child_count": len(children),
        "parents": parents,
        "children": children
    }