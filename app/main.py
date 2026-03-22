# =========================================
# FastAPI 主服务（企业入口）
# =========================================

from fastapi import FastAPI

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


from fastapi import UploadFile, File, HTTPException
import chardet
from core.chunking.parent_child import build_hierarchical_chunks
from core.embedding.embedder import embedder
app = FastAPI()

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
# 接口
# =========================

@app.post("/ingest")
def ingest(text: str):
    """
    写入数据
    """
    # 加入 BM25
    bm25_retriever.add_documents([text])

    result = ingest_pipeline.run(text)

    return result


@app.get("/query")
def query(q: str):
    """
    查询（RAG）
    """
    result = rag_pipeline.run(q)

    return result



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传 .txt 文件，执行父子分片并入库，返回分片详情
    """
    # 1. 校验文件类型
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="仅支持 .txt 文件")

    # 2. 读取并解码文件内容
    try:
        raw = await file.read()
        encoding = chardet.detect(raw)['encoding'] or 'utf-8'
        text = raw.decode(encoding, errors='replace')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件读取失败: {str(e)}")

    # 3. 生成父子块
    try:
        parents, children = build_hierarchical_chunks(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分片失败: {str(e)}")

    print(f"父块数量: {len(parents)}")
    print(f"子块数量: {len(children)}")

    # 4. 将父块存入 MongoDB
    try:
        for parent in parents:
            mongo.save_parent(parent["parent_id"], parent["text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"父块入库失败: {str(e)}")

    # 5. 将子块向量化后存入 Milvus
    try:
        child_texts = [c["chunk"] for c in children]
        child_parent_ids = [c["parent_id"] for c in children]

        print(f"子块文本数量: {len(child_texts)}")
        print(f"父ID数量: {len(child_parent_ids)}")

        # 向量化
        embeddings = embedder.embed(child_texts)  # 保持为 numpy 数组
        print(f"向量数组类型: {type(embeddings)}")
        if hasattr(embeddings, 'shape'):
            print(f"向量数组形状: {embeddings.shape}")
        print(f"向量数量 (len): {len(embeddings)}")

        # 关键检查：确保向量数量与子块数量一致
        if len(embeddings) != len(child_texts):
            raise HTTPException(
                status_code=500,
                detail=f"向量数量({len(embeddings)})与子块数量({len(child_texts)})不一致"
            )

        # 检查向量维度是否与 Milvus 定义一致 (DIM=384)
        if hasattr(embeddings, 'shape') and len(embeddings.shape) == 2:
            vector_dim = embeddings.shape[1]
            print(f"向量维度: {vector_dim}")
            # 注意：Milvus 集合定义的维度是 384，如果你的模型输出是 512，需要调整 Milvus 的 DIM
            # 这里不强制报错，只是打印警告
            if vector_dim != 384:
                print(f"警告：向量维度 {vector_dim} 与 Milvus 定义的 384 不一致，可能导致插入失败")

        # 直接传递 embeddings (numpy 数组) 给 milvus.insert，它会调用 .tolist()
        print("调用 milvus.insert...")
        milvus.insert(embeddings, child_texts, child_parent_ids)
        print("插入成功")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"子块入库失败: {str(e)}\n{traceback.format_exc()}")

    # 6. 返回分片详情
    return {
        "filename": file.filename,
        "message": "文件已处理并入库",
        "parent_count": len(parents),
        "child_count": len(children),
        "parents": parents,
        "children": children
    }