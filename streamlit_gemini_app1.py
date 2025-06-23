import os, time
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Sanity check for API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing GOOGLE_API_KEY in .env")
    st.stop()

@st.cache_resource
def init_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.2,
        max_output_tokens=512,
        max_retries=2,
    )

llm = init_llm()

# Rate limit tracker
RATE_LIMIT_SECONDS = 3.1
if "last_call" not in st.session_state:
    st.session_state.last_call = 0.0

def wait_if_needed():
    elapsed = time.time() - st.session_state.last_call
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)

# Prompt + chain setup
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("human", "{text}")]
)
chain = prompt | llm | StrOutputParser()

# Title
st.title("💬 Gemini 1.5 Pro – LangChain Demo")

# Input row with aligned button
with st.form(key="chat_form", clear_on_submit=False):
    col1, col2 = st.columns([5, 1])
    with col1:
        input_text = st.text_input("Your message:", label_visibility="collapsed")
    with col2:
        send = st.form_submit_button("🔮 Send")
        
# Handle submission
if send and input_text.strip():
    wait_if_needed()
    try:
        with st.spinner("Thinking..."):
            response = chain.invoke({"text": input_text})
        st.markdown(response)
        st.session_state.last_call = time.time()
    except Exception as e:
        st.error("Rate limit hit. Please wait and try again.")
        st.exception(e)
