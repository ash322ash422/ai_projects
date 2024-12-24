from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage


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
    MessagesPlaceholder(variable_name="chat_history"), #added
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
    return response["output"] #added


if __name__ == '__main__':
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(agentExecutor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input)) #added
        chat_history.append(AIMessage(content=response)) #added

        print("Assistant:", response)    #added
"""
You: hello
Assistant: Hello! How can I assist you today?
You: my name is ash
Assistant: Hello Ash! It's great to meet you. How can I help you today?
You: what is my name?
Assistant: Your name is Ash.
You:
"""