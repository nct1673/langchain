import requests
import streamlit as st


def get_openaillm_response(input_text):
    response = requests.post('http://127.0.0.1:8000/openaillm/invoke',
    json = {'input': {'topic': input_text}})

    return response.json()['output']['content']


def get_ollama_response(input_text):
    response = requests.post('http://127.0.0.1:8000/ollama/invoke',
    json = {'input': {'topic': input_text}})

    return response.json()['output']

st.title('Langchain demo with GPT-OSS API')
input_text1 = st.text_input('Tell GPT-3.5-turbo what you curious about')
input_text2 = st.text_input('Tell GPT-OSS what you curious about')


if input_text1:
    st.write(get_openaillm_response(input_text1))

if input_text2:
    st.write(get_ollama_response(input_text2))