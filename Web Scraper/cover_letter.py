import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
import PyPDF2
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer, util

# Page configuration
st.set_page_config(
    page_title="BOTMAN",
    page_icon="🌟",
    layout="centered",
)

llm = ChatGroq(
            temperature=0.1,
            groq_api_key="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",
            model_name="llama-3.1-70b-versatile"
        )
#llm = OllamaLLM(model="llama3.1:8b")

# Function to scrape the job description from a URL
def scrape_job_description(url):
    try:
        # Use the UnstructuredURLLoader for scraping
        loader = UnstructuredURLLoader([url])
        text = loader.load()[0].page_content
        return text
    except Exception as e:
        st.error(f"Error scraping the job description: {e}")
        return None


def calculate_match(jd_text, resume_text):
    # Load pre-trained model (BERT/SBERT)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Compute embeddings for JD and resume
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)

    # Compute cosine similarity between embeddings
    similarity = util.pytorch_cos_sim(jd_embedding, resume_embedding)

    return similarity.item()  # Return similarity score

# Function to generate the cover letter using LLM
def generate_cover_letter(jd_text, resume_text, word):
    # Matching logic (50% or above match)
    match_ratio = calculate_match(jd_text, resume_text)

    # Check if the JD is related to an internship
    if "internship" in jd_text.lower() or "intern" in jd_text.lower():
        # Ignore the matching percentage if it's an internship
        st.info("Internship-related job description detected.")
    elif match_ratio < 0.4:
        # If match ratio is below 50%, reject
        return (f"The match between your job description and resume is insufficient to generate a cover letter.\n"
                f"Current matching ratio: ***{(match_ratio * 100):.2f}%***. \n"
                "Please ensure a minimum matching ratio of ***40%*** between your resume and the job description.")

    st.info(f"The match between the job description and resume is ***{(match_ratio * 100):.2f}%***.")

    # Define the prompt template
    template = """
    Write a formal and professional cover letter tailored to the following job description and resume.

    Your task is to extract and highlight key relevant skills, experiences, and qualifications from the resume that match the job description.

    Be concise and ensure the information from the resume is presented clearly without being overly creative or imaginative. **Make important words bold.**

    Job Description:
    {jd_text}

    Resume:
    {resume_text}

    The cover letter should focus on matching the candidate's experience to the job requirements. 
    Must complete the cover letter in {word} words. Also, write a subject for the email related to the job role.
    """

    # Use PromptTemplate to fill in jd_text, resume_text, and word count
    prompt = PromptTemplate(
        input_variables=["jd_text", "resume_text", "word"],
        template=template
    ).format(jd_text=jd_text, resume_text=resume_text, word=word)

    # Invoke the LLM with the formatted prompt
    cover_letter = llm.invoke(prompt)

    return cover_letter.content


# Function to parse resume PDF
def parse_resume(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        resume_text = ""
        for page in reader.pages:
            resume_text += page.extract_text()
        return resume_text
    except Exception as e:
        st.error(f"Error parsing resume: {e}")
        return None


# Main function for Streamlit app
def main():
    try:
        st.title("Cover Letter Generator")

        # Job Description URL and Resume Upload
        jd_url = st.text_input("Enter Job Description URL:")
        resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

        # Check if JD URL and resume are provided
        if jd_url and resume_file:
            # Scrape JD
            jd_text = scrape_job_description(jd_url)
            if jd_text:
                # Parse resume
                resume_text = parse_resume(resume_file)
                if resume_text:
                    # Input for word limit
                    word = st.text_input("Enter word limit for cover letter:", value="180")

                    # Generate cover letter
                    if st.button("Generate Cover Letter"):
                        cover_letter = generate_cover_letter(jd_text, resume_text, word)
                        if cover_letter:
                            st.write(cover_letter)
    except Exception as e:
        print(e)
        st.write(e)




if __name__ == "__main__":
    main()
