from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
# from langchain_classic import hub # new way
# from langsmith import Client
from langchain_classic.agents import AgentExecutor # new way
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter # New way
from langchain_core.tools import create_retriever_tool


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true" # langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')


# Defining searching tools
# wikipedia
wiki_wrapper = WikipediaAPIWrapper(
    top_k_results= 3,
    doc_content_chars_max= 300
)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# arxiv
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results= 3,
    doc_content_chars_max= 300
)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# web url
loader = WebBaseLoader('https://docs.langchain.com/langsmith/home')
docs = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = text_splitter.split_documents(docs)

vector_db = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector_db.as_retriever()
retriever_tool = create_retriever_tool(retriever, 
                      'langsmith_search',
                      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
                      )



tools = [wiki, arxiv, retriever_tool]

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=3000,
    timeout=30,
    # ... (other params)
)


# 3. Create the agent (This IS the executor now)
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful assistant."
)

# 4. Run it directly (Note the 'messages' format)
result = agent.invoke({
    "messages": [("human", "Tell me about Fourier series, and also tell me you get it from wikipedia or arxiv")]
})

# Access the final response
print(result["messages"][-1].content)