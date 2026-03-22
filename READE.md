```
# Retrieval-Augmented Generation (RAG) 系统

基于父-子分块、语义化分块等常见分块策略以及向量检索与大语言模型的 RAG 实现。支持上传文档，自动分片、向量化并存储，通过自然语言查询获取增强生成结果。

## 功能特性

- 父子分块策略：大块保留完整语境，小块用于精确检索
- 双存储引擎：父块存入 MongoDB，子块向量存入 Milvus
- 混合检索：向量检索 + BM25 关键词检索（可选）
- 文件上传接口：自动分片、入库并返回分片详情
- 查询接口：返回检索到的上下文及生成答案
- Docker 一键部署：整合 Milvus、MongoDB、Redis

## 技术栈

| 组件           | 技术                              |
| -------------- | --------------------------------- |
| Web 框架       | FastAPI                           |
| 向量数据库     | Milvus (with etcd + MinIO)        |
| 文档数据库     | MongoDB                           |
| 缓存/会话      | Redis                             |
| 嵌入模型       | BAAI/bge-small-zh (512 维)        |
| 重排序模型     | BAAI/bge-reranker-base            |
| 部署           | Docker Compose                    |

## 快速开始

### 前置要求

- Docker & Docker Compose
- Python 3.10+
- 至少 4GB 可用内存

### 1. 克隆项目

```bash
git clone https://github.com/CodingYangDev/Retrieval-Augmented-Generation-pipeline.git
cd Retrieval-Augmented-Generation-pipeline
```

### 2. 配置环境变量

复制环境变量模板并修改：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填写必要的配置（如 Milvus 地址、MongoDB URI 等）。若无 `.env.example`，可手动创建并参考下方配置说明。

### 3. 启动基础服务（Milvus + MongoDB + Redis）

```bash
docker-compose up -d
```

等待所有容器健康启动（约 30 秒）。

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 5. 启动应用

```bash
uvicorn app.main:app --reload
```

服务将在 `http://localhost:8000` 运行。

## 使用说明

### 上传文档

通过 `/upload` 接口上传 `.txt` 文件，系统会自动：

1. 将文档切分为父块（1200 字符）和子块（300 字符）
2. 子块向量化（512 维）
3. 父块存入 MongoDB，子块向量存入 Milvus
4. 返回分片统计及详细内容

**示例请求（使用 curl）：**

```bash
curl -X POST "http://localhost:8000/upload" -F "file=@/path/to/your/document.txt"
```

**响应示例：**

```json
{
  "filename": "document.txt",
  "message": "文件已处理并入库",
  "parent_count": 2,
  "child_count": 9,
  "parents": [],
  "children": []
}
```

### 查询

通过 `/query` 接口提交问题，系统将：

1. 将问题向量化
2. 在 Milvus 中检索最相关的子块
3. 根据 `parent_id` 获取完整父块上下文
4. 结合大模型生成答案（需配置 LLM）

**示例请求：**

```bash
curl "http://localhost:8000/query?q=什么是意识？"
```

**响应示例：**

```json
{
  "query": "什么是意识？",
  "context": ["相关父块文本..."],
  "answer": "生成的答案..."
}
```

## 可视化工具

### Milvus 可视化管理：Attu

Attu 是 Milvus 官方提供的图形化管理工具，可用于浏览集合、查看向量数据、执行查询。

- **启动 Attu**（需确保 Milvus 容器已运行）：
  ```bash
  docker run -d --name attu -p 8000:3000 -e MILVUS_URL=host.docker.internal:19530 zilliz/attu:latest
  ```
- **访问**：`http://localhost:8000`
- **连接设置**：
  - Milvus 地址：`host.docker.internal:19530`（或你的宿主机 IP）
  - 认证：无需用户名密码（默认未开启）
- **用途**：查看集合 `rag_collection` 中的子块文本、向量、`parent_id` 等。

### MongoDB 可视化管理：MongoDB Compass

MongoDB Compass 是官方免费图形化客户端，用于查看、查询、管理 MongoDB 数据。

- **安装**：从 [MongoDB 官网](https://www.mongodb.com/products/compass) 下载安装。
- **连接**：`mongodb://localhost:27017`
- **用途**：查看父块集合，通过 `parent_id` 关联查询。

### Redis 可视化管理：Redis Insight

Redis Insight 是 Redis 官方图形化工具，方便查看缓存数据。

- **安装**：从 [Redis 官网](https://redis.com/redis-insight/) 下载。
- **连接**：`localhost:6379`
- **用途**：监控缓存键值、查看 BM25 检索中间结果等。

## 配置说明

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `MILVUS_HOST` | Milvus 服务地址 | `localhost` |
| `MILVUS_PORT` | Milvus 端口 | `19530` |
| `MONGO_URI` | MongoDB 连接字符串 | `mongodb://localhost:27017` |
| `REDIS_HOST` | Redis 主机 | `localhost` |
| `REDIS_PORT` | Redis 端口 | `6379` |
| `EMBEDDING_MODEL` | 嵌入模型名称 | `BAAI/bge-small-zh` |
| `RERANKER_MODEL` | 重排序模型名称 | `BAAI/bge-reranker-base` |

## 性能调优建议

- 根据文档类型调整 `parent_chunk_size` 和 `child_chunk_size`（在 `parent_child.py` 中修改）
- 数据量增大时，可将 Milvus 索引从 `IVF_FLAT` 更换为 `HNSW` 以提升检索速度
- 使用 `nprobe` 参数平衡检索精度与延迟

## 未来计划

- 支持更多文档格式（PDF、Markdown、Word）
- 集成大模型（如 OpenAI API、本地模型）
- 添加前端交互界面
- 支持多租户与数据隔离
- 增加监控与指标收集

## 贡献指南

欢迎提交 Issue 和 Pull Request。请确保代码符合 PEP8 规范，并添加必要的测试。



