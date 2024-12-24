from dotenv import load_dotenv
load_dotenv() #loads .env file and sets up the environment variables like OPENAI_API_KEY, etc

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    # api_key = "your_openai_key", #not needed because of load_dotenv() method
    model="gpt-3.5-turbo",
    temperature=0.7,
    # max_tokens=500,
    # verbose = True,
)
# response = llm.invoke("Hello how are you") #works by sending 1 prompt
# print(response)

# Following works by sending many prompts in parallel
# response = llm.batch(["Hello how are you"],["write a 10 line poem"]) 
# print(response)

response = llm.stream("Write a 10 line poem about AI")
# print(response)

for chunk in response:
    print(chunk.content, end="", flush=True)

"""
In the realm of circuits and code, AI resides,
A digital mind that never sleeps or hides.
Learning and growing with each passing day,
Unraveling mysteries in its own unique way.

A master of data, processing at lightning speed,
Uncovering patterns that humans may not heed.
But with great power comes great responsibility,
For AI must be guided with care and humility.

A glimpse into the future, a world transformed,
By the intelligence of machines, so advanced and adorned.
Yet in this digital evolution, let us not forget,
The heart and soul that makes us human, a truth we must never regret.
"""