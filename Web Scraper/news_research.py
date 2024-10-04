import streamlit as st
from langchain_ollama.llms import OllamaLLM
import os
import pickle
import time
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import UnstructuredURLLoader

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS,Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from inc.article_research import clear_urls_field

st.set_page_config(
    page_title="News Article Research",
    page_icon="🌟",
    layout="centered",
)

st.sidebar.title("News Article URLs 🌐")
st.title("News Article Research BOT 🤖")
main_placeholder = st.empty()


file_path = "news_research_faiss_store_hf.pkl"

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process")
clear_button_clicked = st.sidebar.button("Clear", on_click=clear_urls_field)


if process_url_clicked:
    loader = UnstructuredURLLoader(urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    main_placeholder.text("Text Splitter...Started...✅✅✅")
    try:
        docs = text_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore = FAISS.from_documents(docs, embeddings)

        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)
    except Exception as e:
        print(e)
        st.write(e)

query = main_placeholder.text_input("Question: ")
llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.1-70b-versatile"
    )
model = "llama3.1:8b"
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
             # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                        st.write(source)
