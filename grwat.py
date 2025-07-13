import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("groq_api_key")

# Initialize Groq client
client = Groq(api_key=api_key)

# Set Streamlit page config
st.set_page_config(page_title="Groq WhatsApp Chatbot", page_icon="")

# Custom CSS to mimic WhatsApp styling
st.markdown("""
<style>
.stChatMessage {
    border-radius:12px;
    padding:10px16px;
    margin-bottom:8px;
    max-width:70%;
    word-break: break-word;
    font-size:1.05em;
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
""", unsafe_allow_html=True)

# App title
st.title(" Groq WhatsApp Chatbot")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Input form at the bottom
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", "")
    submit_button = st.form_submit_button(label="Send")

    # Handle user input
    if submit_button and user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Call the Groq LLM API
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=st.session_state.messages,
            temperature=0.13,
            max_completion_tokens=681,
            top_p=1,
            stop=None,
        )
        assistant_reply = response.choices[0].message.content

        # Append assistant message
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        # Rerun to show new messages
        st.experimental_rerun()

# Display the chat conversation
for msg in reversed(st.session_state.messages[1:]):
    # skip system prompt
    if msg["role"] == "user":
        st.markdown(
            f'<div class="stChatMessage user-msg">You: {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    elif msg["role"] == "assistant":
        st.markdown(
            f'<div class="stChatMessage assistant-msg">Assistant: {msg["content"]}</div>',
            unsafe_allow_html=True
        )   