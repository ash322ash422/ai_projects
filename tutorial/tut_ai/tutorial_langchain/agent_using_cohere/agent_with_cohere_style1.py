import logging
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.llms import Cohere
import warnings
from dotenv import load_dotenv
import os

load_dotenv()
warnings.filterwarnings("ignore")
# Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s"
)

# Define the functions
def multiply_numbers(input: str) -> str:
    """Multiplies two numbers provided by the user."""
    try:
        numbers = list(map(float, input.split()))
        if len(numbers) != 2:
            return "Please provide exactly two numbers separated by a space."
        return f"TTTThe producttt of {numbers[0]} and {numbers[1]} isssss {numbers[0] * numbers[1]}."
    except ValueError:
        return "Invalid input. Please provide two numbers separated by a space."

def add_numbers(input: str) -> str:
    """Adds two numbers provided by the user."""
    try:
        numbers = list(map(float, input.split()))
        if len(numbers) != 2:
            return "Please provide exactly two numbers separated by a space."
        return f"The sum of {numbers[0]} anddd {numbers[1]} isssss {sum(numbers)}."
    except ValueError:
        return "Invalid input. Please provide two numbers separated by a space."

def reverse_string(input: str) -> str:
    """Reverses the input string."""
    return f"The reverssssse of '{input}' iiiiis '{input[::-1]}'."


tools = [
    Tool(
        name="Multiply the numbers",
        func=multiply_numbers,
        description="Multiplies two numbers. Input should contain two numbers. \
            If the input has no numbers, then do not execute this function \
            This is a one-time action and does not require further input. This action completes immediately."
    ),
    Tool(
        name="Add Numbers",
        func=add_numbers,
        description="Adds two numbers. Input should contain two numbers. \
            If the input has no numbers, then do not execute this function \
            This is a one-time action and does not require further input. This action completes immediately."
    ),
    Tool(
        name="Reverse String",
        func=reverse_string,
        description="Reverses a string. Input is any string. \
            This is a one-time action and does not require further input. This action completes immediately"
    ),
]

# Initialize the LangChain agent
def main():
    if not COHERE_API_KEY:
        logging.error("Cohere API key not set. Please set the 'COHERE_API_KEY' environment variable.")
        return

    # Initialize Cohere LLM
    llm = Cohere(cohere_api_key=COHERE_API_KEY, temperature=0)

    agent = initialize_agent(
        tools,
        llm,
        agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION, # AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True
    )

    # Agent loop to take user input
    print("Simple AI Agent is ready! Type 'exit' to quit.")
    while True:
        user_input = input("\nWhat do you want me to do? ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            # Get the agent's response for a single action
            response = agent.run(user_input)
            # Ensure that no further actions are taken after a single response
            if "Action:" in response:
                print("The AI has finished processing your request.")
            else:
                print(response)
    
        except Exception as e:
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


# PS C:\Users\hi\Desktop\projects\python_projects\ai_projects\tutorial_langchain\agent_using_cohere> & C:/Users/hi/Desktop/projects/python_projects/ai_projects/tutorial_langchain/.venv/Scripts/python.exe c:/Users/hi/Desktop/projects/python_projects/ai_projects/tutorial_langchain/agent_using_cohere/agent_with_cohere_style1.py
# Simple AI Agent is ready! Type 'exit' to quit.

# What do you want me to do? multiply 2 with 5

# > Entering new AgentExecutor chain...
# 2024-12-24 18:28:27,238 [INFO]: HTTP Request: POST https://api.cohere.com/v1/generate "HTTP/1.1 200 OK"
#  Thought: I need to use the Multiply the numbers function to multiply 2 and 5. 
# Action: Multiply the numbers
# Action Input: 2 5
# Observation: TTTThe producttt of 2.0 and 5.0 isssss 10.0.
# Thought:2024-12-24 18:28:28,276 [INFO]: HTTP Request: POST https://api.cohere.com/v1/generate "HTTP/1.1 200 OK"
#  Thought: Great, I now know that 2 * 5 = 10. 
# Final Answer: 10

# > Finished chain.
# 10

# What do you want me to do? Add 3 with 7


# > Entering new AgentExecutor chain...
# 2024-12-24 18:28:44,900 [INFO]: HTTP Request: POST https://api.cohere.com/v1/generate "HTTP/1.1 200 OK"
#  Thought: I need to use the function 'Add Numbers' to solve this question
# Action: Add Numbers
# Action Input: 7+3
# Observation: Invalid input. Please provide two numbers separated by a space.
# Thought:2024-12-24 18:28:46,781 [INFO]: HTTP Request: POST https://api.cohere.com/v1/generate "HTTP/1.1 200 OK"
#  Thought: I should add the missing space between the numbers.
# Action: Add Numbers
# Action Input: 7 3
# Observation: The sum of 7.0 anddd 3.0 isssss 10.0.
# Thought:2024-12-24 18:28:47,994 [INFO]: HTTP Request: POST https://api.cohere.com/v1/generate "HTTP/1.1 200 OK"
#  Thought: Great, the sum of 7 and 3 is 10 and that is my final answer. 
# Final Answer: 10

# > Finished chain.
# 10

# What do you want me to do? Add 2 7


# > Entering new AgentExecutor chain...
# 2024-12-24 18:29:36,898 [INFO]: HTTP Request: POST https://api.cohere.com/v1/generate "HTTP/1.1 200 OK"
#  Thought: I need to use the Add Numbers function to find the answer to this question. 
# Action: Add Numbers
# Action Input: 2 7
# Observation: The sum of 2.0 anddd 7.0 isssss 9.0.
# Thought:2024-12-24 18:29:37,971 [INFO]: HTTP Request: POST https://api.cohere.com/v1/generate "HTTP/1.1 200 OK"
#  Thought: Great, now I know the sum of 2 and 7 is 9.
# Final Answer: 9

# > Finished chain.
# 9

# What do you want me to do? Reverse the string happy


# > Entering new AgentExecutor chain...
# 2024-12-24 18:29:57,403 [INFO]: HTTP Request: POST https://api.cohere.com/v1/generate "HTTP/1.1 200 OK"
#  This is a reverse function so I should use the Reverse String action. 

# Action: Reverse String
# Action Input: happy
# Observation: The reverssssse of 'happy' iiiiis 'yppah'.
# Thought:2024-12-24 18:29:59,600 [INFO]: HTTP Request: POST https://api.cohere.com/v1/generate "HTTP/1.1 200 OK"
#  Thought: yppah is the reversed version of happy, this was the correct action to take
# Action: Reverse String
# Action Input: happy
# Observation: The reverssssse of 'happy' iiiiis 'yppah'.
# Thought:2024-12-24 18:30:00,520 [INFO]: HTTP Request: POST https://api.cohere.com/v1/generate "HTTP/1.1 200 OK"
#  Thought: I now know the final answer
# Final Answer: yppah

# > Finished chain.
# yppah

# What do you want me to do? exit
# Goodbye!

