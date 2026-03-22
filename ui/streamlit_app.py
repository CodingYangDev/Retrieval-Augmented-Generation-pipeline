import streamlit as st
import requests

st.title("🚀 企业级 RAG 系统")

query = st.text_input("请输入你的问题")

if query:
    res = requests.get(
        "http://localhost:8000/query",
        params={"q": query}
    ).json()

    st.subheader("🧠 回答")
    st.write(res["answer"])

    st.subheader("📚 召回文档")
    for doc in res["docs"]:
        st.write(doc)