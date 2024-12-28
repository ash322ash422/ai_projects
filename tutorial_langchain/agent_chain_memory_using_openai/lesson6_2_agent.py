from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

#Give agent an intruction, and it will determine which actions to take and which sequence
# Agents are able to interact with their env while using tools

# Here we create 2 tools for agent: 1st tool will allow agent to search the internet for
# answer, 2nd tool will grant the agent access to our custom knowledge base

# Create Retriever
loader = WebBaseLoader("https://python.langchain.com/docs/expression_language/")
docs = loader.load()
    
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splitDocs = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
vectorStore = FAISS.from_documents(docs, embedding=embedding)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

model = ChatOpenAI(
    model='gpt-3.5-turbo-1106',
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant called Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

search = TavilySearchResults()
retriever_tools = create_retriever_tool(
    retriever,
    name="lcel_search",
    description="Use this tool when searching for information about Langchain Expression Language (LCEL)."
)
tools = [search, retriever_tools]

agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
)

def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response["output"]

if __name__ == '__main__':
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(agentExecutor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Assistant:", response)

#The code works but receive following error. Solution is trim chat_history or summarize chat_history
"""
raise self._make_status_error_from_response(err.response) from None
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 16385 tokens. However, your messages resulted in 16990 tokens (16877 in the messages, 113 in the functions). Please reduce the length of the messages or functions.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}

"""