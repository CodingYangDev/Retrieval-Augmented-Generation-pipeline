import streamlit as st
import requests

st.title("🚀 企业级 RAG 系统")

import streamlit as st
import requests
import json

# 后端服务地址（请根据实际情况修改）
API_BASE_URL = "http://localhost:8001"

st.set_page_config(page_title="RAG 系统管理后台", layout="wide")
st.title("📚 RAG 系统管理后台")

# 侧边栏导航
menu = st.sidebar.radio("功能", ["📂 文件上传与切片", "🔍 智能查询"])

# ========== 文件上传与切片 ==========
if menu == "📂 文件上传与切片":
    st.header("上传文档并执行父子切片")
    uploaded_file = st.file_uploader("选择 .txt 文件", type=["txt"])

    if uploaded_file is not None:
        # 显示文件信息
        st.info(f"文件名: {uploaded_file.name} | 大小: {uploaded_file.size} 字节")

        if st.button("🚀 执行切片并入库", type="primary"):
            with st.spinner("正在处理文件，请稍候..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
                try:
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ 处理成功！父块数量: {data['parent_count']}，子块数量: {data['child_count']}")

                        # 展示父块
                        with st.expander(f"📦 父块列表（共 {data['parent_count']} 个）"):
                            for i, parent in enumerate(data['parents'], 1):
                                st.markdown(f"**父块 {i}** (ID: `{parent['parent_id']}`)")
                                st.text_area(f"内容 {i}", parent['text'], height=150, key=f"parent_{i}")
                                st.markdown("---")

                        # 展示子块
                        with st.expander(f"🧩 子块列表（共 {data['child_count']} 个）"):
                            for i, child in enumerate(data['children'], 1):
                                st.markdown(f"**子块 {i}** (父ID: `{child['parent_id']}`)")
                                st.text(child['chunk'])
                                st.markdown("---")
                    else:
                        st.error(f"❌ 处理失败：{response.text}")
                except Exception as e:
                    st.error(f"❌ 请求异常：{str(e)}")

    # 侧边栏使用说明
    with st.sidebar.expander("📖 使用说明"):
        st.markdown("""
        - 支持上传 `.txt` 文本文件。
        - 点击按钮后，后端将自动执行父子切片（父块约1200字符，子块约300字符）。
        - 父块会存入 MongoDB，子块向量存入 Milvus。
        - 页面会展示切片结果，方便验证分片效果。
        """)

# ========== 智能查询 ==========
elif menu == "🔍 智能查询":
    st.header("智能问答")
    query = st.text_input("请输入您的问题", placeholder="例如：什么是人工智能？")

    if query:
        with st.spinner("思考中..."):
            try:
                response = requests.get(f"{API_BASE_URL}/query", params={"q": query})
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("💡 回答")

                    # 尝试提取常见字段，若没有则显示原始 JSON
                    if "answer" in data:
                        st.markdown(f"> {data['answer']}")
                    elif "result" in data:
                        st.markdown(f"> {data['result']}")
                    else:
                        st.json(data)

                    # 展示召回文档（如果后端返回了 docs 字段）
                    docs = data.get("docs") or data.get("documents") or data.get("context")
                    if docs:
                        st.subheader("📚 相关文档片段")
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**片段 {i}**")
                            st.text(doc)
                            st.markdown("---")
                else:
                    st.error(f"查询失败：{response.text}")
            except Exception as e:
                st.error(f"请求异常：{str(e)}")