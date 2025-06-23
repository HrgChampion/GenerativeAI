import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Set up LangChain ChatOpenAI with streaming
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo",
    streaming=True
)

# Streamlit app UI
st.title("ChatGPT Clone with LangChain")

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input from user
if prompt := st.chat_input("What is up?"):
    # Store user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert history to LangChain message objects
    langchain_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        else:
            langchain_messages.append(AIMessage(content=msg["content"]))

    # Stream assistant response
    with st.chat_message("assistant"):
        stream = llm.stream(langchain_messages)
        full_response = st.write_stream(stream)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
