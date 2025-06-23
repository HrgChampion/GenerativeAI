import streamlit as st
import time
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()

st.title("Document Search with LangChain")

urls = ['https://www.victoriaonmove.com.au/local-removalists.html','https://victoriaonmove.com.au/index.html','https://victoriaonmove.com.au/contact.html']

loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

all_splits = docs
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings(),
    persist_directory="chroma_db"
)
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 6})

llm = OpenAI(temperature=0.4, max_tokens=512)\

query = st.chat_input("Enter your query here:")
prompt = query

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)



response = rag_chain.invoke({"input": query})
st.write(response['answer'])