from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import (
    UpstashRedisChatMessageHistory,
)
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

#Following program stores and retrieves chat history from persistent storage
# - Goto uptash.com, create an account, then create database  under Redis 'langchainpython', then scroll down to 'Rest API' section and
#    get UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN
UPSTASH_URL="UPSTASH_REDIS_REST_URL here"
UPSTASH_TOKEN="UPSTASH_REDIS_REST_TOKEN here"

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.6
)

prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly AI assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])


history = UpstashRedisChatMessageHistory(
    url=UPSTASH_URL, token=UPSTASH_TOKEN,
    ttl=500, # conversation  expires after 500 sec. Time To Live
    session_id="chat1" #id of this session. Can be linked to user profile.
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history,
)

# chain = prompt | model
chain = LLMChain(
    llm=model,
    prompt=prompt,
    verbose=True,
    memory=memory
)


# Prompt 1
q1 = { "input": "My name is Leon" }
resp1 = chain.invoke(q1)
print(resp1["text"])

# Prompt 2
q2 = { "input": "What is my name?" }
resp2 = chain.invoke(q2)
print(resp2["text"])

"""
c:\Users\hi\Desktop\projects\python_projects\ai_projects\tutorial_langchain\lesson7_memory.py:40: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
  memory = ConversationBufferMemory(
c:\Users\hi\Desktop\projects\python_projects\ai_projects\tutorial_langchain\lesson7_memory.py:47: LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
  chain = LLMChain(


> Entering new LLMChain chain...
Prompt after formatting:
System: You are a friendly AI assistant.
Human: My name is Leon

> Finished chain.
Hello Leon! How can I assist you today?


> Entering new LLMChain chain...
Prompt after formatting:
System: You are a friendly AI assistant.
Human: My name is Leon
AI: Hello Leon! How can I assist you today?
Human: What is my name?

> Finished chain.
Your name is Leon.
"""