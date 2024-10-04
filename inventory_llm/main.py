import streamlit as st
from dotenv import load_dotenv
from langchain_helper import get_few_shot_db_chain,temp
import os
import pymysql
# how many color do you have?

load_dotenv()
st.set_page_config(
    page_title="Inventory QnA"
)
st.title(os.environ["APP_NAME"])

question = st.text_input("Write your question : ")


if question:
    print(question)
    chain = get_few_shot_db_chain()
    response = chain.run(question)

    st.header("Answer")
   
    sql_query = response.split("SQLQuery:")[1].split("SQLResult:")[0].strip()
    st.write(sql_query)
    # Execute the SQL query directly on the database
    connection = pymysql.connect(
        host="localhost",
        user="root",
        password="password",
        database="bandhu_battery"
    )

    with connection.cursor() as cursor:
        cursor.execute(sql_query)
        sql_result = cursor.fetchall()

    # Display the SQL result
    st.write("SQL Result:")
    st.code(sql_result)







# give me in tebular format in todays sells product name