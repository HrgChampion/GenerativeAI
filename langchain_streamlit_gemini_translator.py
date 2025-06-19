import getpass
import os,time
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.2,
    max_output_tokens=512,
    max_retries=0,
)

prompt =ChatPromptTemplate.from_messages(
    [("system",
      "You are a helpful assistant that translates text from English to French."), 
     ("human", "{text}")]
)

st.title("💬 Gemini LangChain Translator Demo")
input_text = st.text_input("Enter text in English to translate to French:")

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke(
        {
        "input_language": "English",
        "output_language": "French",
        "text": input_text
        }
    ))