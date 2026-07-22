"""Streamlit demo for the PDF RAG question-answering system."""

import hashlib
from pathlib import Path

import streamlit as st

from run import PDFQA


APP_TITLE = "PDF 智能阅读问答 Demo"
SAMPLE_PDF = Path("data/files/sample_document.pdf")
UPLOAD_DIR = Path("temp/uploads")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="PDF",
    layout="wide",
    initial_sidebar_state="expanded",
)


def save_uploaded_pdf(uploaded_file) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    data = uploaded_file.getbuffer()
    digest = hashlib.md5(data).hexdigest()[:12]
    target = UPLOAD_DIR / f"{digest}-{safe_name}"
    target.write_bytes(data)
    return str(target)


def get_qa(pdf_source: str, api_key: str) -> PDFQA:
    """Keep the QA instance isolated to the current browser session."""
    identity = (pdf_source, api_key)
    if st.session_state.get("qa_identity") != identity:
        st.session_state.qa = PDFQA(pdf_source=pdf_source, api_key=api_key)
        st.session_state.qa_identity = identity
    return st.session_state.qa


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.last_pdf_source = None


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_pdf_source" not in st.session_state:
    st.session_state.last_pdf_source = None
if "last_api_key" not in st.session_state:
    st.session_state.last_api_key = None


with st.sidebar:
    st.header("配置")
    api_key = st.text_input(
        "DashScope API Key",
        value="",
        type="password",
        help="仅用于当前浏览器会话，不会写入项目文件或部署配置。",
    ).strip()
    if api_key:
        st.success("API Key 已就绪")
    else:
        st.warning("请输入 DashScope API Key")

    source_mode = st.radio("PDF 来源", ["使用示例 PDF", "上传 PDF", "PDF URL"], index=0)

    pdf_source = ""
    if source_mode == "使用示例 PDF":
        pdf_source = str(SAMPLE_PDF)
        st.caption(str(SAMPLE_PDF))
    elif source_mode == "上传 PDF":
        uploaded = st.file_uploader("选择 PDF 文件", type=["pdf"])
        if uploaded is not None:
            pdf_source = save_uploaded_pdf(uploaded)
            st.caption(pdf_source)
    else:
        pdf_source = st.text_input("PDF URL", placeholder="https://example.com/file.pdf")

    if st.button("清空对话", use_container_width=True):
        reset_chat()


st.title(APP_TITLE)
st.caption("上传或选择一个 PDF，然后基于文档内容进行连续问答。")

if not api_key:
    st.info("请在左侧输入 DashScope API Key 后再开始。")
    st.stop()

if st.session_state.last_api_key != api_key:
    reset_chat()
    st.session_state.last_api_key = api_key

if not pdf_source:
    st.info("请选择一个 PDF 来源。")
    st.stop()

if st.session_state.last_pdf_source != pdf_source:
    st.session_state.messages = []
    st.session_state.last_pdf_source = pdf_source

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("evidence"):
            with st.expander("查看引用证据"):
                for index, item in enumerate(message["evidence"], 1):
                    page = item.page + 1 if item.page is not None else "未知"
                    st.markdown(f"**[{index}] 来源：{item.source}，页码：{page}，相关性：{item.score:.1f}**")
                    st.write(item.content)

question = st.chat_input("向当前 PDF 提问")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            qa = get_qa(pdf_source, api_key)
            with st.spinner("正在检索并生成答案，首次提问会先构建索引..."):
                result = qa.ask(question)

            answer = result.get("answer", "")
            st.markdown(answer)

            evidence = result.get("evidence", [])
            if evidence:
                with st.expander("查看引用证据"):
                    for index, item in enumerate(evidence, 1):
                        page = item.page + 1 if item.page is not None else "未知"
                        st.markdown(
                            f"**[{index}] 来源：{item.source}，页码：{page}，相关性：{item.score:.1f}**"
                        )
                        st.write(item.content)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "evidence": evidence}
            )
        except Exception as exc:
            error_message = f"运行失败：{exc}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
