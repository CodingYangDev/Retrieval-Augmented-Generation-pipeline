<template>
  <div class="app-container">
    <el-card class="main-card" shadow="never">

      <!-- 头部 -->
      <div class="header">
        <div>
          <h1 class="title">RAG 智能系统</h1>
          <div class="subtitle">文档解析 · 切片 · 检索</div>
        </div>
      </div>

      <el-tabs v-model="activeTab">

        <!-- 上传与切片 Tab -->
        <el-tab-pane label="上传与切片" name="upload">
          <div class="upload-layout">

            <!-- 左侧配置 -->
            <div class="config-panel">
              <div class="panel-title">分块策略</div>

              <el-radio-group v-model="chunkMethod">
                <el-radio value="parent_child">父子</el-radio>
                <el-radio value="fixed">固定</el-radio>
                <el-radio value="semantic">语义</el-radio>
                <el-radio value="hybrid">混合</el-radio>
              </el-radio-group>

              <div class="divider"></div>

              <!-- 父子模式 -->
              <div v-if="chunkMethod==='parent_child' || chunkMethod==='hybrid'">
                <div class="param">
                  <span>父块大小</span>
                  <el-input-number v-model="parentSize"/>
                </div>
                <div class="param">
                  <span>父块重叠</span>
                  <el-input-number v-model="parentOverlap"/>
                </div>
                <div class="param">
                  <span>子块大小</span>
                  <el-input-number v-model="childSize"/>
                </div>
                <div class="param">
                  <span>子块重叠</span>
                  <el-input-number v-model="childOverlap"/>
                </div>
              </div>

              <!-- 固定模式 -->
              <div v-if="chunkMethod==='fixed' || chunkMethod==='hybrid'">
                <div class="param">
                  <span>固定大小</span>
                  <el-input-number v-model="fixedSize"/>
                </div>
                <div class="param">
                  <span>固定重叠</span>
                  <el-input-number v-model="fixedOverlap"/>
                </div>
              </div>

              <!-- 语义模式 -->
              <div v-if="chunkMethod==='semantic'">
                <span>相似度</span>
                <el-slider v-model="similarityThreshold"/>
              </div>

            </div>

            <!-- 右侧上传区域 -->
            <div class="upload-panel">
              <el-upload
                drag
                :auto-upload="false"
                :on-change="handleFileChange"
                :limit="1"
                accept=".txt,.pdf,.docx,.pptx,.md,.xlsx"
              >
                <div>拖拽文件或点击上传</div>
              </el-upload>

              <el-button
                type="primary"
                :loading="uploading"
                @click="handleUpload"
                :disabled="!selectedFile"
              >
                开始处理
              </el-button>

              <div v-if="uploadResult">
                <p>父块: {{ uploadResult.parent_count }}</p>
                <p>子块: {{ uploadResult.child_count }}</p>
              </div>

            </div>

          </div>
        </el-tab-pane>

        <!-- 查询 Tab -->
        <el-tab-pane label="查询" name="query">
          <div class="query-box">
            <el-input v-model="queryText" type="textarea" :rows="3"/>
            <el-button type="primary" @click="handleQuery">查询</el-button>

            <div v-if="queryResult">
              <p>{{ queryResult.answer }}</p>
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

const API_BASE_URL = 'http://localhost:8001'

const activeTab = ref('upload')

const chunkMethod = ref('parent_child')
const parentSize = ref(1200)
const parentOverlap = ref(200)
const childSize = ref(300)
const childOverlap = ref(50)
const fixedSize = ref(500)
const fixedOverlap = ref(50)
const similarityThreshold = ref(0.6)

const selectedFile = ref(null)
const uploading = ref(false)
const uploadResult = ref(null)

const queryText = ref('')
const queryResult = ref(null)

const handleFileChange = (file) => {
  selectedFile.value = file.raw
}

const handleUpload = async () => {
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
  } else if (chunkMethod.value === 'fixed') {
    params = {
      chunk_method: 'fixed',
      chunk_size: fixedSize.value,
      overlap: fixedOverlap.value,
    }
  } else if (chunkMethod.value === 'semantic') {
    params = {
      chunk_method: 'semantic',
      similarity_threshold: similarityThreshold.value,
    }
  } else if (chunkMethod.value === 'hybrid') {
    params = {
      chunk_method: 'hybrid',
      parent_size: parentSize.value,
      parent_overlap: parentOverlap.value,
      child_size: childSize.value,
      child_overlap: childOverlap.value,
      chunk_size: fixedSize.value,
      overlap: fixedOverlap.value,
    }
  }

  formData.append('params', JSON.stringify(params))

  uploading.value = true
  try {
    const res = await axios.post(`${API_BASE_URL}/upload`, formData)
    uploadResult.value = res.data
    ElMessage.success('成功')
  } catch (e) {
    ElMessage.error('失败')
  } finally {
    uploading.value = false
  }
}

const handleQuery = async () => {
  const res = await axios.get(`${API_BASE_URL}/query`, {
    params: { q: queryText.value }
  })
  queryResult.value = res.data
}
</script>

<style scoped>
.app-container {
  background: #f5f7fa;
  padding: 20px;
}

.main-card {
  border-radius: 16px;
  box-shadow: 0px 6px 16px rgba(0, 0, 0, 0.1);
}

.header {
  padding: 16px;
  border-bottom: 1px solid #ececec;
}

.title {
  font-size: 24px;
  font-weight: bold;
}

.subtitle {
  font-size: 14px;
  color: #a0a0a0;
}

.upload-layout {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 24px;
}

.config-panel {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
}

.panel-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 10px;
}

.divider {
  height: 1px;
  background-color: #f0f0f0;
  margin: 16px 0;
}

.param {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
}

.upload-panel {
  background: #ffffff;
  border-radius: 8px;
  padding: 16px;
}

.upload-panel .el-upload {
  border: 2px dashed #d1d5db;
  padding: 20px;
  border-radius: 8px;
  background-color: #fafafa;
}

.upload-panel .el-button {
  margin-top: 16px;
}

.query-box {
  max-width: 700px;
  margin: 40px auto;
  padding: 20px;
  background: #fff;
  border-radius: 8px;
  border: 1px solid #d1d5db;
}

.query-box .el-input,
.query-box .el-button {
  margin-top: 12px;
}

.query-box .el-button {
  width: 120px;
}
</style>
