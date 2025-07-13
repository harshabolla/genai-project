import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()
client = Groq(api_key=os.getenv("groq_api_key"))

st.set_page_config(page_title="Groq WhatsApp Chatbot", page_icon="💬")
st.markdown(
    """
    <style>
    .stChatMessage {
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 8px;
        max-width: 70%;
        word-break: break-word;
        font-size: 1.1em;
    }
    .user-msg {
        background-color: #dcf8c6;
        margin-left: auto;
        text-align: right;
    }
    .assistant-msg {
        background-color: #fff;
        margin-right: auto;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 Groq WhatsApp Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant in coding ."}
    ]

# Display conversation in WhatsApp style
for msg in st.session_state.messages[1:]:  # skip system prompt
    if msg["role"] == "user":
        st.markdown(
            f'<div class="stChatMessage user-msg">You: {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    elif msg["role"] == "assistant":
        st.markdown(
            f'<div class="stChatMessage assistant-msg">Assistant: {msg["content"]}</div>',
            unsafe_allow_html=True,
        )

# User input at the bottom
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message...", key="input")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=st.session_state.messages,
        temperature=0.13,
        max_completion_tokens=681,
        top_p=1,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    st.session_state.messages.append({"role": "assistant", "content": response})