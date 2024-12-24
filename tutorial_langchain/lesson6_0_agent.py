from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool


# - Give agent an instruction, and it will determine which actions to take and which sequence
# - Agents are able to interact with their env while using tools
# - Goto Tavily AI website and get TAVILY_API_KEY

# Here we create 1 tools for agent: allow agent to search the internet for answer

model = ChatOpenAI(
    model='gpt-3.5-turbo-1106',
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant called Max."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

search = TavilySearchResults()
tools = [search, ]

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
    return response

if __name__ == '__main__':
    chat_history = []
    user_input = "What is the current weather in Johannesburg ?"
    response = process_chat(agentExecutor, user_input, chat_history)
    print("response=",response)
    
"""
response= {'input': 'What is the current weather in Johannesburg ?', 'chat_history': [], 'output': 'The current weather in Johannesburg is clear with a temperature of 18.3°C (64.9°F). The wind speed is 5.0 kph and the humidity is at 56%.'}
"""