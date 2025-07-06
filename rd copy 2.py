import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chains import RetrievalQAWithSourcesChain

# Load environment variables
load_dotenv()

# UI Setup
st.set_page_config(page_title="HarshaBot - Agentic Q&A", layout="wide")
st.title("🤖 HarshaBot: Agentic AI for PDF/Word Docs")
st.sidebar.title("📤 Upload Multiple Documents")

# File upload
uploaded_files = st.sidebar.file_uploader("Choose PDF or Word files", type=["pdf", "docx"], accept_multiple_files=True)
process = st.sidebar.button("📥 Process Files")
query = st.text_input("💬 Ask a question about the documents:")
VECTORSTORE_PATH = "faiss_agent_index"

# Initialize LLM and Embeddings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
st.sidebar.success("✅ Gemini LLM initialized")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Process uploaded documents
if process and uploaded_files:
    all_docs = []
    with st.spinner("🔄 Processing documents..."):
        try:
            os.makedirs("temp", exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join("temp", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                if uploaded_file.name.lower().endswith(".pdf"):
                    try:
                        loader = PyMuPDFLoader(file_path)
                        #docs = loader.load()
                        if not docs or len(docs[0].page_content) < 10:  # Minimum content check
                            raise ValueError("PDF appears to be empty or image-based")
                    except Exception as pdf_error:
                        st.warning(f"⚠ PyMuPDF failed for {uploaded_file.name}, trying unstructured fallback...")
                        loader = UnstructuredFileLoader(file_path, mode="elements", strategy="fast")
                        #docs = loader.load()
                else:
                    loader = UnstructuredFileLoader(file_path)

                docs = loader.load()
                all_docs.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(all_docs)

            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(VECTORSTORE_PATH)
            st.success("✅ Documents processed and indexed!")

        except Exception as e:
            st.error(f"❌ Error processing files: {e}")

# Run query through agent
if query:
    if os.path.exists(VECTORSTORE_PATH):
        with st.spinner("🧠 Thinking agentically..."):
            try:
                vectorstore = FAISS.load_local(
                    VECTORSTORE_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True
                )

                # Define retrieval tool
                tools = [
                    Tool.from_function(
                        func=lambda q: "\n".join(
                            [doc.page_content for doc in vectorstore.similarity_search(q, k=3)]
                        ),
                        name="Document Retriever",
                        description="Useful for answering questions about the uploaded documents"
                    )
                ]

                agent = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True
                )

                answer = agent.run(query)
                st.subheader("🤖 Agentic Answer:")
                st.write(answer)

            except Exception as e:
                st.error(f"❌ Agent error: {e}")
    else:
        st.warning("⚠ Please upload and process files first.")
