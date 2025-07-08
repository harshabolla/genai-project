import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

# Load API key from .env
load_dotenv()

# Streamlit UI
st.title("🔍 HarshaBot: News Research Tool (Gemini + FAISS)")
st.sidebar.title("Enter Article URLs")

# Sidebar inputs
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process = st.sidebar.button("Process URLs")

# Placeholder
query = st.text_input("Ask a question about the articles:")
main_placeholder = st.empty()

# Set paths
VECTORSTORE_PATH = "faiss_index"



llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
print("LLM initialized.",llm)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Process URLs
if process:
    with st.spinner("🔄 Loading and indexing articles..."):
        try:
            loader = UnstructuredURLLoader(urls=[url for url in urls if url])
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
            )
            split_docs = text_splitter.split_documents(docs)

            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(VECTORSTORE_PATH)

            st.success("✅ Articles processed and indexed.")
        except Exception as e:
            st.error(f"Error processing URLs: {e}")

# Handle Query
if query:
    if os.path.exists(VECTORSTORE_PATH):
        try:
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

            qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )

            result = qa_chain({"question": query})

            st.subheader("🧠 Answer:")
            st.write(result["answer"])

            if result.get("sources"):
                st.subheader("📚 Sources:")
                for source in result["sources"].split("\n"):
                    st.write(source)

        except Exception as e:
            st.error(f"Query error: {e}")
    else:
        st.warning("⚠ Please process URLs first.")
