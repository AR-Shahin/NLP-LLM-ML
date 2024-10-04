import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(
    page_title="Your Website Title",
    page_icon="🌟",
    layout="centered",
)
st.title("Website BOT")

model = "llama3.1:8b" # llama3:70b | llama3:latest
llm = OllamaLLM(model=model)

r = llm.invoke("Hi")

st.write(r)

