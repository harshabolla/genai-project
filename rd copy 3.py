import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

# Load environment variables
load_dotenv()

# Page setup
st.set_page_config(page_title="HarshaBot Chat", layout="wide")
st.title("💬 HarshaBot: Intelligent Chat with Your Documents")
st.sidebar.title("📤 Upload PDFs or Word Docs")

# Session state for chat and agent
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent" not in st.session_state:
    st.session_state.agent = None

# File upload
uploaded_files = st.sidebar.file_uploader("Upload your documents", type=["pdf", "docx"], accept_multiple_files=True)
process = st.sidebar.button("📥 Process Files")
VECTORSTORE_PATH = "faiss_agent_index"

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Document processing
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
                        docs = loader.load()
                    except Exception:
                        loader = UnstructuredFileLoader(file_path, mode="elements", strategy="fast")
                        docs = loader.load()
                else:
                    loader = UnstructuredFileLoader(file_path)
                    docs = loader.load()

                all_docs.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(all_docs)

            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(VECTORSTORE_PATH)
            st.success("✅ Documents processed and indexed!")

            # Load and setup agent
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
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
                verbose=False
            )
            st.session_state.agent = agent

        except Exception as e:
            st.error(f"❌ Error processing files: {e}")

# Chat UI (WhatsApp-style)
if st.session_state.agent:
    for role, message in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(message)

    user_input = st.chat_input("Ask something about the documents...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        try:
            with st.spinner("🤖 HarshaBot is typing..."):
                response = st.session_state.agent.run(user_input)
            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("bot", response))
        except Exception as e:
            st.error(f"❌ Agent error: {e}")
else:
    st.info("👈 Upload and process documents to start chatting.")

