from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true" # langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant, please response to user's query"),
        ("user", "Question: {question}")
    ]
)

# streamlit framework
st.title('Langchain demo with OpenAI API')
input_text = st.text_input("Search the topic you want")

# OpenAI LLM
llm = Ollama(model='gpt-oss:20b')
output_parser =  StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))