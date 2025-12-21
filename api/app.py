from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama

from dotenv import load_dotenv


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true" # langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

app = FastAPI(
    title= 'Langchain Server',
    version='1.0',
    description= 'A simple API Server'
)

add_routes(
    app,
    ChatOpenAI(),
    path= '/openai'
)



model = ChatOpenAI()
llm = Ollama(model='gpt-oss:20b')


prompt1 = ChatPromptTemplate.from_template('Introduce {topic} with approximately 200 words')
prompt2 = ChatPromptTemplate.from_template('Introduce {topic} with approximately 1000 words')


add_routes(
    app,
    prompt1 | model ,
    path= '/openaillm' 
)

add_routes(
    app,
    prompt1 | llm ,
    path= '/ollama' 
)


if __name__ == '__main__':
    uvicorn.run(
        app,
        host='127.0.0.1',
        port=8000
    )





