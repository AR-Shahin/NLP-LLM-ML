import os
import streamlit as st
import pickle
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory  # Import the memory class
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
st.set_page_config(
    page_title="BOTMAN",
    page_icon="🌟",
    layout="centered",
)
st.title("BOTMAN 🤖")
st.sidebar.title("Enter Your site URL")

url = st.sidebar.text_input("URL")
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"

main_placeholder = st.empty()
llm = ChatGroq(
    temperature=0.2,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile"
)

# Create a memory object
memory = ConversationBufferMemory(memory_key="chat_history")

def get_all_links(base_url):
    print(f"Base URL is: {base_url}")
    try:
        response = requests.get(base_url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        links = set()

        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            # Ensure the link is a full URL
            if link.startswith('/'):
                link = base_url + link
            elif not link.startswith('http'):
                continue
            links.add(link)
        print(f"Total Links found: {len(links)}")
        return links
    except Exception as e:
        print(f"Error fetching {base_url}: {e}")
        return set()

if process_url_clicked:
    # Load data
    main_placeholder.text("Scraping your site 🌐")
    urls = get_all_links(url)
    print(f"Total {len(urls)}")
    print(f"Links are {list(urls)}")
    st.write(f"Total pages found: {len(urls)}")
    loader = UnstructuredURLLoader(urls=list(urls))
    main_placeholder.text("Data Loading... Started...✅✅✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter... Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save it to FAISS index
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    main_placeholder.text("Ask your question! ")

query = st.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            main_placeholder.text("Generate your response...✅✅✅")
            vectorstore = pickle.load(f)

            # Create the chain without output_key
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                output_key="answer"  # Specify output_key explicitly
            )

            # Create the input dictionary for the chain
            input_data = {"question": query}

            # Get the result
            result = chain(input_data)

            # Extract answer and sources manually
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            main_placeholder.text("Here is your response...✅✅✅")
            st.header("Answer")
            st.write(answer)  # Accessing the answer directly

            # Display sources, if available
            if sources:
                st.subheader("Sources:")
                for source in sources:  # Iterate through sources
                    st.write(source)

            # Update memory after processing the query
            memory.chat_memory.append({"question": query, "answer": answer})
