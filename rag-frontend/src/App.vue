<template>
  <div class="app-container">
    <el-card class="main-card" shadow="never">
      <div class="header">
        <h1 class="title">📚 RAG 智能系统</h1>
        <span class="subtitle">Powered by FastAPI + Milvus</span>
      </div>

      <el-tabs v-model="activeTab" type="border-card" class="tabs">
        <!-- 上传与切片 Tab -->
        <el-tab-pane label="📂 上传与切片" name="upload">
          <div class="upload-layout">
            <div class="config-panel">
              <h3>⚙️ 分块参数</h3>

              <div class="method-row">
                <div class="method-label">分块策略：</div>
                <div class="method-options">
                  <el-radio-group v-model="chunkMethod">
                    <el-radio value="parent_child">父子模式</el-radio>
                    <el-radio value="semantic">语义化分块</el-radio>
                    <el-radio value="fixed">固定token分块</el-radio>
                  </el-radio-group>
                </div>
              </div>

              <!-- 父子模式参数 -->
              <div v-if="chunkMethod === 'parent_child'" class="params-group">
                <div class="param-item">
                  <span class="param-label">父块大小(字符)</span>
                  <el-input-number v-model="parentSize" :min="200" :max="5000" :step="100" controls-position="right" style="width: 100%" />
                </div>
                <div class="param-item">
                  <span class="param-label">父块重叠(字符)</span>
                  <el-input-number v-model="parentOverlap" :min="0" :max="1000" :step="50" controls-position="right" style="width: 100%" />
                </div>
                <div class="param-item">
                  <span class="param-label">子块大小(字符)</span>
                  <el-input-number v-model="childSize" :min="50" :max="1000" :step="50" controls-position="right" style="width: 100%" />
                </div>
                <div class="param-item">
                  <span class="param-label">子块重叠(字符)</span>
                  <el-input-number v-model="childOverlap" :min="0" :max="200" :step="10" controls-position="right" style="width: 100%" />
                </div>
              </div>

              <!-- 语义化分块参数 -->
              <div v-if="chunkMethod === 'semantic'" class="params-group">
                <div class="param-item">
                  <span class="param-label">相似度阈值</span>
                  <el-slider v-model="similarityThreshold" :min="0" :max="1" :step="0.05" show-input />
                  <div class="param-hint">值越大，分块越细碎；值越小，分块越粗糙</div>
                </div>
              </div>

              <!-- 固定token分块参数 -->
              <div v-if="chunkMethod === 'fixed'" class="params-group">
                <div class="param-item">
                  <span class="param-label">块大小(token)</span>
                  <el-input-number v-model="fixedSize" :min="100" :max="2000" :step="50" controls-position="right" style="width: 100%" />
                </div>
                <div class="param-item">
                  <span class="param-label">重叠(token)</span>
                  <el-input-number v-model="fixedOverlap" :min="0" :max="500" :step="10" controls-position="right" style="width: 100%" />
                </div>
              </div>
            </div>

            <div class="upload-panel">
              <h3>📎 上传文件</h3>
              <el-upload
                class="upload-area"
                drag
                :auto-upload="false"
                :on-change="handleFileChange"
                :limit="1"
                accept=".txt"
              >
                <el-icon class="upload-icon"><UploadFilled /></el-icon>
                <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
                <template #tip>
                  <div class="el-upload__tip">仅支持 .txt 文本文件</div>
                </template>
              </el-upload>

              <el-button
                type="primary"
                :loading="uploading"
                @click="handleUpload"
                :disabled="!selectedFile"
                class="upload-btn"
              >
                🚀 执行切片并入库
              </el-button>

              <div v-if="uploadResult" class="result-section">
                <el-alert
                  :title="`✅ 处理成功！父块数量: ${uploadResult.parent_count}，子块数量: ${uploadResult.child_count}`"
                  type="success"
                  show-icon
                  class="mb-4"
                />
                <el-collapse v-model="activeCollapse">
                  <el-collapse-item title="📦 父块列表" name="parents">
                    <div v-for="parent in uploadResult.parents" :key="parent.parent_id" class="chunk-item">
                      <div class="chunk-meta">ID: {{ parent.parent_id }}</div>
                      <div class="chunk-text">{{ parent.text }}</div>
                    </div>
                  </el-collapse-item>
                  <el-collapse-item title="🧩 子块列表" name="children">
                    <div v-for="(child, idx) in uploadResult.children" :key="idx" class="chunk-item">
                      <div class="chunk-meta">父ID: {{ child.parent_id }}</div>
                      <div class="chunk-text">{{ child.chunk }}</div>
                    </div>
                  </el-collapse-item>
                </el-collapse>
              </div>
            </div>
          </div>
        </el-tab-pane>

        <!-- 智能问答 Tab -->
        <el-tab-pane label="🔍 智能问答" name="query">
          <div class="query-container">
            <div class="retrieval-selector">
              <span class="label">检索方式：</span>
              <el-radio-group v-model="retrievalMethod">
                <el-radio value="hybrid">混合检索（向量 + BM25）</el-radio>
                <el-radio value="vector">仅向量检索</el-radio>
                <el-radio value="bm25">仅 BM25 检索</el-radio>
              </el-radio-group>
            </div>

            <el-input
              v-model="queryText"
              type="textarea"
              :rows="3"
              placeholder="请输入您的问题..."
              class="query-input"
            />

            <div class="query-actions">
              <el-button type="primary" :loading="querying" @click="handleQuery" size="large">
                🔍 查询
              </el-button>
            </div>

            <div v-if="queryResult" class="query-result">
              <el-card shadow="never" class="answer-card">
                <template #header>
                  <div class="card-header">💡 回答</div>
                </template>
                <div class="answer-text">{{ queryResult.answer || queryResult.result || '暂无答案' }}</div>
              </el-card>

              <div v-if="queryResult.docs || queryResult.documents || queryResult.context">
                <h4 class="docs-title">📚 相关文档片段</h4>
                <div
                  v-for="(doc, idx) in (queryResult.docs || queryResult.documents || queryResult.context)"
                  :key="idx"
                  class="chunk-item"
                >
                  <div class="chunk-text">{{ doc }}</div>
                </div>
              </div>
            </div>
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'

const API_BASE_URL = 'http://localhost:8001'

const activeTab = ref('upload')

const chunkMethod = ref('parent_child')
const parentSize = ref(1200)
const parentOverlap = ref(200)
const childSize = ref(300)
const childOverlap = ref(50)
const similarityThreshold = ref(0.6)
const fixedSize = ref(500)
const fixedOverlap = ref(50)

const selectedFile = ref(null)
const uploading = ref(false)
const uploadResult = ref(null)
const activeCollapse = ref(['parents', 'children'])

const retrievalMethod = ref('hybrid')
const queryText = ref('')
const querying = ref(false)
const queryResult = ref(null)

const handleFileChange = (file) => {
  selectedFile.value = file.raw
}

const handleUpload = async () => {
  if (!selectedFile.value) {
    ElMessage.warning('请先选择文件')
    return
  }

  uploading.value = true
  uploadResult.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)

    let params = {}
    if (chunkMethod.value === 'parent_child') {
      params = {
        chunk_method: 'parent_child',
        parent_size: parentSize.value,
        parent_overlap: parentOverlap.value,
        child_size: childSize.value,
        child_overlap: childOverlap.value,
      }
    } else if (chunkMethod.value === 'semantic') {
      params = {
        chunk_method: 'semantic',
        similarity_threshold: similarityThreshold.value,
      }
    } else {
      params = {
        chunk_method: 'fixed',
        chunk_size: fixedSize.value,
        overlap: fixedOverlap.value,
      }
    }

    formData.append('params', JSON.stringify(params))

    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })

    if (response.status === 200) {
      uploadResult.value = response.data
      ElMessage.success('切片入库成功')
    } else {
      ElMessage.error(`上传失败：${response.data.detail || '未知错误'}`)
    }
  } catch (error) {
    console.error(error)
    ElMessage.error(error.response?.data?.detail || '请求异常')
  } finally {
    uploading.value = false
  }
}

const handleQuery = async () => {
  if (!queryText.value.trim()) {
    ElMessage.warning('请输入问题')
    return
  }

  querying.value = true
  queryResult.value = null

  try {
    const response = await axios.get(`${API_BASE_URL}/query`, {
      params: {
        q: queryText.value,
        retrieval_method: retrievalMethod.value,
      },
    })

    if (response.status === 200) {
      queryResult.value = response.data
    } else {
      ElMessage.error(`查询失败：${response.data.detail || '未知错误'}`)
    }
  } catch (error) {
    console.error(error)
    ElMessage.error(error.response?.data?.detail || '请求异常')
  } finally {
    querying.value = false
  }
}
</script>

<style scoped>
/* 样式与之前相同，无变化，此处省略（用户已有） */
</style>

<style scoped>
.app-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 24px;
}

.main-card {
  border-radius: 24px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid #eef2f6;
}

.title {
  font-size: 28px;
  font-weight: 700;
  background: linear-gradient(120deg, #667eea, #764ba2);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  margin: 0;
}

.subtitle {
  font-size: 14px;
  color: #6b7280;
}

.tabs {
  margin-top: 0;
}

.upload-layout {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 32px;
  padding: 24px;
}

.config-panel h3,
.upload-panel h3 {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 20px;
  color: #1f2937;
}

.method-row {
  display: flex;
  align-items: center;
  margin-bottom: 24px;
  flex-wrap: wrap;
  gap: 12px;
}

.method-label {
  font-weight: 500;
  color: #374151;
  min-width: 70px;
}

.method-options {
  flex: 1;
}

.params-group {
  margin-top: 8px;
  margin-bottom: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.param-item {
  width: 100%;
}

.param-label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  color: #374151;
  margin-bottom: 8px;
}

.param-hint {
  font-size: 12px;
  color: #9ca3af;
  margin-top: 4px;
}

.upload-area {
  border: 2px dashed #e2e8f0;
  border-radius: 20px;
  background: #fafbfc;
  transition: all 0.2s;
  margin-bottom: 20px;
}

.upload-area:hover {
  border-color: #667eea;
  background: #f5f3ff;
}

.upload-icon {
  font-size: 48px;
  color: #667eea;
}

.upload-btn {
  width: 100%;
  margin-top: 8px;
}

.result-section {
  margin-top: 24px;
}

.chunk-item {
  background: #f8fafc;
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 12px;
  border-left: 3px solid #cbd5e1;
  transition: all 0.2s;
}

.chunk-item:hover {
  border-left-color: #667eea;
  background-color: #fefce8;
}

.chunk-meta {
  font-family: monospace;
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 6px;
}

.chunk-text {
  font-size: 14px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

.query-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 24px;
}

.retrieval-selector {
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.retrieval-selector .label {
  font-weight: 500;
  color: #374151;
}

.query-input {
  margin-bottom: 20px;
}

.query-actions {
  text-align: center;
  margin-bottom: 32px;
}

.query-result {
  margin-top: 24px;
}

.answer-card {
  background: linear-gradient(120deg, #e0e7ff, #f9fafb);
  border-left: 5px solid #667eea;
  margin-bottom: 24px;
}

.card-header {
  font-weight: 600;
  font-size: 18px;
}

.answer-text {
  font-size: 16px;
  line-height: 1.6;
  color: #1f2937;
  white-space: pre-wrap;
}

.docs-title {
  font-weight: 600;
  font-size: 18px;
  margin: 20px 0 12px 0;
  color: #1f2937;
}
</style>
