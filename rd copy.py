import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from tempfile import NamedTemporaryFile

# Load API key
load_dotenv()

# Streamlit app setup
st.set_page_config(page_title="HarshaBot - Upload PDF/Word", layout="wide")
st.title("🧠 HarshaBot: RAG from PDF/Word (Gemini + FAISS)")

# Upload files
uploaded_files = st.file_uploader("📤 Upload PDF or Word files", type=["pdf", "docx"], accept_multiple_files=True)
process = st.button("📥 Process Files")
query = st.text_input("💬 Ask a question based on the documents:")
main_placeholder = st.empty()

VECTORSTORE_PATH = "faiss_index"

# Initialize LLM and embeddings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=1024,
    max_retries=2,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

@st.cache_resource
def create_or_load_vectorstore(docs):
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

if process:
    if not uploaded_files:
        st.warning("⚠ Please upload at least one file.")
    else:
        with st.spinner("⏳ Processing files..."):
            try:
                all_docs = []
                for file in uploaded_files:
                    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[-1]) as temp_file:
                        temp_file.write(file.read())
                        temp_path = temp_file.name

                    if file.name.endswith(".pdf"):
                        loader = PyPDFLoader(temp_path)
                    elif file.name.endswith(".docx"):
                        loader = UnstructuredWordDocumentLoader(temp_path)
                    else:
                        continue

                    docs = loader.load()
                    all_docs.extend(docs)

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                split_docs = splitter.split_documents(all_docs)
                vectorstore = create_or_load_vectorstore(split_docs)
                st.success("✅ Files processed and vectorstore created.")

            except Exception as e:
                st.error(f"❌ Error: {e}")

if query:
    if os.path.exists(VECTORSTORE_PATH):
        with st.spinner("🔍 Generating answer..."):
            try:
                vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

                qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    return_source_documents=True
                )

                result = qa_chain.invoke({"question": query})
                st.subheader("💡 Answer:")
                st.write(result["answer"])

                if result.get("sources"):
                    st.subheader("📚 Sources:")
                    for src in result["sources"].split("\n"):
                        st.write(src)

            except Exception as e:
                st.error(f"❌ Query error: {e}")
    else:
        st.warning("⚠ Please process documents first.")
