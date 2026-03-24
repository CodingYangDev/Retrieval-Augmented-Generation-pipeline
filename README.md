# Retrieval-Augmented Generation (RAG) 系统

> 一个完整的 RAG 实现，支持多种分块策略、查询重写、混合检索、重排序与提示词工程。上传 PDF、Word、Markdown 等文档，自动解析、向量化，通过自然语言查询获取增强生成结果。

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-compose-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://codingyangdev.github.io/Retrieval-Augmented-Generation-pipeline/)

---

## 📌 项目介绍

做文档问答时，我们通常会遇到这些难题：

- **文档怎么切**才能既保留完整语境又保证检索精度？
- **用户问题表达模糊**时，如何让检索更准确？
- **检索到的片段**如何筛选排序，选出最相关的内容？
- **怎么设计提示词**让大模型生成更可靠的答案？

这个项目提供了一套完整的 RAG 解决方案，覆盖从文档处理到答案生成的每一个关键环节：

- **灵活的分块策略**：固定 Token、父子分块、语义分块，按需选择
- **查询理解与重写**：改写用户问题，提升检索命中率
- **混合检索**：向量检索 + BM25 关键词检索，取长补短
- **重排序 (Rerank)**：对召回结果二次排序，筛选最相关内容
- **提示词工程**：精心设计的提示词模板，引导大模型生成高质量回答

支持 txt、pdf、doc/docx、markdown 等常见文档格式，上传后自动解析、分块、向量化并存入 Milvus 和 MongoDB，通过 HTTP 接口进行问答。

> 📚 **完整教程请点击这里**：[在线阅读](https://codingyangdev.github.io/Retrieval-Augmented-Generation-pipeline/) —— 无需下载，随时随地学习

---

## ✨ 你将收获什么？

- 📖 **开源免费**：MIT 协议，随意使用和学习
- 🧩 **多种分块策略**：深入理解不同分块方式的原理与适用场景
- ✍️ **查询重写技术**：学会如何改写用户问题以提升检索效果
- 🔍 **混合检索实现**：掌握向量检索 + BM25 的融合策略
- 🎯 **重排序优化**：使用 BGE-reranker 等模型提升相关性
- 🧠 **提示词工程**：设计高质量提示词，让大模型生成更准确的答案
- 🏗️ **亲手部署**：从零开始部署一套完整的 RAG 系统
- 📄 **多格式文档解析**：掌握 PDF、Word、Markdown 等文档的解析技巧
- 🚀 **实战驱动**：通过 API 快速集成到自己的应用中

---

## 🚀 快速开始

### 前置要求
- Docker & Docker Compose
- Python 3.10+
- 至少 4GB 可用内存

### 1. 克隆项目
```bash
git clone https://github.com/CodingYangDev/Retrieval-Augmented-Generation-pipeline.git
cd Retrieval-Augmented-Generation-pipeline
```

### 2. 启动依赖服务
```bash
docker-compose up -d
```

### 3. 安装依赖并运行
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问 `http://localhost:8000/docs` 查看 API 文档。

> 详细的部署步骤、环境变量配置、生产环境部署等，请查看[完整教程](https://codingyangdev.github.io/Retrieval-Augmented-Generation-pipeline/)。

---

## 📚 教程导航

我们的[在线教程](https://codingyangdev.github.io/Retrieval-Augmented-Generation-pipeline/)包含以下内容，建议按顺序学习：

| 章节 | 内容简介 |
| --- | --- |
| **快速开始** | 环境配置、Docker 启动、运行第一个查询 |
| **分块策略详解** | 固定 Token、父子分块、语义分块的原理与选型建议 |
| **文档解析支持** | PDF、Word、Markdown、TXT 的解析实现 |
| **MongoDB 使用指南** | 父块存储、文档管理、索引优化 |
| **Milvus 向量数据库使用** | 向量集合管理、索引构建、检索优化 |
| **查询理解与重写** | 如何改写用户问题以提升检索效果 |
| **混合检索与重排序** | 向量检索、BM25、RRF 融合排序、重排序模型应用 |
| **大模型调用与提示词工程** | 提示词模板设计、模型调用、答案生成 |
| **API 调用示例** | curl 和 Python 调用上传、查询接口的完整示例 |
| **自定义扩展** | 自定义分块策略、自定义检索器 |
| **生产环境部署** | Docker 单独部署、性能调优、监控配置 |

---

## 🛠 技术栈

| 组件 | 技术 |
| --- | --- |
| Web 框架 | FastAPI |
| 向量数据库 | Milvus (with etcd + MinIO) |
| 文档数据库 | MongoDB |
| 缓存/会话 | Redis |
| 文档解析 | PyPDF2, python-docx, markdown |
| 嵌入模型 | BAAI/bge-small-zh (512 维) |
| 重排序模型 | BAAI/bge-reranker-base |
| 检索策略 | 向量检索 + BM25 + RRF 融合 |
| 大模型 | OpenAI API / 本地模型（可扩展） |
| 部署 | Docker Compose |

---

## 🤝 如何贡献

欢迎任何形式的贡献！如果你有新的分块策略想尝试，或者发现了 bug，欢迎提 PR 或 Issue。

1. Fork 这个仓库
2. 创建你的功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的改动 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个 Pull Request

如果你不确定如何修改，可以先开一个 Issue 讨论。

---




如果这个项目对你有帮助，欢迎点个 ⭐️ Star 支持一下～
```

---

**修改说明**：
1. **项目介绍**部分加入了“查询理解与重写”“重排序”“提示词工程”的描述，突出全流程。
2. **你将收获什么**列表增加了查询重写、混合检索、重排序、提示词工程等条目。
3. **教程导航**表格中，“查询理解与重写”“混合检索与重排序”“大模型调用与提示词工程”独立成行，明确展示。
4. **技术栈**补充了检索策略（向量+BM25+RRF）和大模型选项。
