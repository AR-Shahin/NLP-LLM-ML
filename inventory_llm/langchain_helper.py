from langchain_groq import ChatGroq
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # Ensure this is correct for your version
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
import os
from dotenv import load_dotenv
from few_shorts import few_shots

load_dotenv()

api_key = os.environ["API_KEY"]

def get_few_shot_db_chain():
    try:
        # Database connection details
        db_user = "root"
        db_password = "password"
        db_host = "localhost"
        db_name = "bandhu_battery"

        # Create SQL database connection
        db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                                  sample_rows_in_table_info=3)

        # Initialize LLM
        llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="llama-3.1-70b-versatile"
        )

        # Define embeddings
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Prepare texts and metadata
        to_vectorize = [" ".join(str(value) for value in example.values()) for example in few_shots]
        metadatas = [
            {k: str(v) for k, v in example.items()}  # Ensure metadata is a dict with simple values
            for example in few_shots
        ]

        # Create vector store
        vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=metadatas)

        # Define example selector
        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vectorstore,
            k=2,
        )

        # Define prompt templates
        mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
            Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
            Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
            Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            Pay attention to use CURDATE() function to get the current date, if the question involves "today".
            
            make sure dont give any in field list is ambiguous
            Use the following format:

            Question: Question here
            SQLQuery: Query to run with no pre-amble
            SQLResult: Result of the SQLQuery
            Answer: Final answer here

            No pre-amble.
            """

        example_prompt = PromptTemplate(
            input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
            template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
        )

        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=mysql_prompt,
            suffix=PROMPT_SUFFIX,
            input_variables=["input", "table_info", "top_k"],
        )

        # Create and return the SQLDatabaseChain
        chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
        return chain

    except Exception as e:
        # Print the exception to understand what went wrong
        print(f"An error occurred: {e}")
        return None


def temp():
    db_user = "root"
    db_password = "password"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)

    llm = ChatGroq(
        temperature=0,
        groq_api_key=api_key,
        model_name="llama-3.1-70b-versatile"
    )

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".

    Use the following format:

    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here

    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer", ],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}\nShahin",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],  # These variables are used in the prefix and suffix
    )
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)

    return chain