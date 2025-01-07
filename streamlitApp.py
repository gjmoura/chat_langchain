import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import os

os.environ["OPENAI_API_KEY"] = "sua-chave-open-api"


st.title("💬 Dados Abertos App")

mysql_uri = 'mysql+mysqlconnector://root:12345@localhost:3307/dados_abertos'

db = SQLDatabase.from_uri(mysql_uri)


def generate_response(input_text):
    answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
 
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    llm = ChatOpenAI()

    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer_prompt
        | llm
        | StrOutputParser()
    )
    
    st.info(chain.invoke({"question": input_text}))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if not os.environ["OPENAI_API_KEY"].startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted:
        generate_response(text)