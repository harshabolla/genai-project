import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

load_dotenv()

# --- WhatsApp-style Header ---
st.markdown("""
<div style='
    background-color: #075E54;
    color: white;
    padding: 12px 16px;
    border-radius: 8px 8px 0 0;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0;
'>
    <img src='https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg' width='36' style='border-radius:50%;background:white;padding:2px;' />
    <span style='font-size: 22px; font-weight: bold;'>HarshaBot</span>
</div>
""", unsafe_allow_html=True)

# --- WhatsApp-style CSS ---
st.markdown("""
<style>
body {
    background-color: #ECE5DD;
}
.block-container {
    padding-top: 0rem;
}
.chat-container {
    background-color: #ECE5DD;
    border-radius: 0 0 8px 8px;
    padding: 10px;
    height: 500px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    border: 1px solid #CCC;
    margin-bottom: 0;
}
.message {
    max-width: 70%;
    padding: 10px 15px;
    margin: 4px;
    border-radius: 8px;
    font-size: 16px;
    word-wrap: break-word;
    line-height: 1.4;
}
.user {
    background-color: #DCF8C6;
    align-self: flex-end;
    text-align: right;
    border: 1px solid #B2D8A8;
}
.bot {
    background-color: #FFFFFF;
    align-self: flex-start;
    text-align: left;
    border: 1px solid #DDD;
}
.input-container {
    display: flex;
    gap: 8px;
    margin-top: 0;
    position: sticky;
    bottom: 0;
    background: #ECE5DD;
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Page setup
st.set_page_config(page_title="HarshaBot WhatsApp Chat", layout="wide")

# Sidebar
st.sidebar.title("📤 Upload PDFs or Word Docs")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.sidebar.file_uploader(
    "Upload your documents", type=["pdf", "docx"], accept_multiple_files=True
)
process = st.sidebar.button("📥 Process Files")
VECTORSTORE_PATH = "faiss_agent_index"

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
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
                        loader = UnstructuredLoader(file_path)
                        docs = loader.load()
                else:
                    loader = UnstructuredLoader(file_path)
                    docs = loader.load()

                all_docs.extend(docs)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = splitter.split_documents(all_docs)

            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(VECTORSTORE_PATH)
            st.success("✅ Documents processed and indexed!")
        except Exception as e:
            st.error(f"❌ Error processing files: {e}")

# Load agent
if os.path.exists(VECTORSTORE_PATH) and "agent" not in st.session_state:
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
            description="Answers questions about the uploaded documents"
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    st.session_state.agent = agent

# ...existing code...

# --- Chat messages ---
chat_html = "<div class='chat-container' id='chatbox'>"
for role, msg in st.session_state.chat_history:
    if role == "user":
        chat_html += f"<div class='message user'>{msg}</div>"
    else:
        chat_html += f"<div class='message bot'>{msg}</div>"
# Add spinner as the last "message" if typing
if st.session_state.get("is_typing", False):
    chat_html += "<div class='message bot'><i>🤖 HarshaBot is typing...</i></div>"
chat_html += "</div>"

st.markdown(chat_html, unsafe_allow_html=True)



# --- Chat input fixed at bottom ---
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
col1, col2 = st.columns([8, 1])

if "input_value" not in st.session_state:
    st.session_state.input_value = ""

def send_message():
    user_input = st.session_state.input_value
    if user_input:
        st.session_state.is_typing = True
        with st.spinner("🤖 HarshaBot is typing..."):
            try:
                response = st.session_state.agent.run(user_input)
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", response))
                st.session_state.input_value = ""
                st.session_state.is_typing = False
                st.rerun()  # Only here!
            except Exception as e:
                st.session_state.is_typing = False
                st.error(f"❌ Agent error: {e}")

with col1:
    st.text_input(
        "Type your message...",
        key="input_value",
        placeholder="Type your message here...",
        label_visibility="collapsed",
        on_change=send_message
    )
with col2:
    st.button("Send", on_click=send_message)

st.markdown("</div>", unsafe_allow_html=True)

# --- Auto-scroll chat to bottom using JS ---
# --- Auto-scroll chat to bottom using JS ---
st.markdown("""
<script>
window.addEventListener('load', function() {
    var chatbox = document.getElementById('chatbox');
    if (chatbox) { chatbox.scrollTop = chatbox.scrollHeight; }
});
var chatbox = document.getElementById('chatbox');
if (chatbox) { chatbox.scrollTop = chatbox.scrollHeight; }
</script>
""", unsafe_allow_html=True)