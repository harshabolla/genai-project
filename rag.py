import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

# Load env vars
load_dotenv()

# UI config
st.set_page_config(page_title="💬 HarshaBot WhatsApp UI", layout="centered")
st.title("💬 HarshaBot: Chat With Your Documents")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload documents
st.sidebar.title("📂 Upload PDFs or Word Docs")
uploaded_files = st.sidebar.file_uploader("Choose files", type=["pdf", "docx"], accept_multiple_files=True)
process = st.sidebar.button("📥 Process Documents")

# Constants
VECTORSTORE_PATH = "faiss_agent_index"

# Gemini Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Document processing
if process and uploaded_files:
    all_docs = []
    with st.spinner("🔄 Processing uploaded files..."):
        try:
            os.makedirs("temp", exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join("temp", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                if uploaded_file.name.endswith(".pdf"):
                    try:
                        loader = PyMuPDFLoader(file_path)
                        docs = loader.load()
                    except:
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

            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

            tools = [
                Tool.from_function(
                    func=lambda q: "\n".join([doc.page_content for doc in vectorstore.similarity_search(q, k=3)]),
                    name="Document Retriever",
                    description="Use this to answer questions about the uploaded documents"
                )
            ]

            st.session_state.agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False
            )

            st.success("✅ Documents processed and HarshaBot is ready!")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Show chat bubbles
if "agent" in st.session_state:
    for role, message in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(message)

    user_query = st.chat_input("Ask a question...")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        try:
            with st.spinner("🤖 HarshaBot is typing..."):
                answer = st.session_state.agent.run(user_query)
            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.chat_history.append(("user", user_query))
            st.session_state.chat_history.append(("bot", answer))
        except Exception as e:
            st.error(f"❌ HarshaBot error: {e}")
else:
    st.info("👈 Upload documents and click **Process Documents** to activate HarshaBot.")
