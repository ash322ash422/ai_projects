# 1) Download and install Ollama from https://ollama.com/
# 2) Pull the latest model: ` ollama pull llama3.2:latest `
# 3) Run the model: ` ollama run llama3.2:latest ` or ` ollama serve `

import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/generate" # Make sure ollama is running locally
MODEL_NAME = "llama3.2" #this is right

def get_response(prompt, model=MODEL_NAME):
    try:
        # Construct the payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True  # Enable streaming response
        }

        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            if response.status_code == 200:
                # Stream the response and concatenate the chunks
                output = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        output += chunk
                return output
            else:
                return f"Error: {response.status_code} - {response.text}"

    except requests.ConnectionError:
        return "Error: Unable to connect to the Ollama server. Ensure Ollama is running."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def main():
    print("\nWelcome to the Local LLM Console Application (Powered by Ollama)")
    print("Type 'exit' to quit the application.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = get_response(user_input)
        
        print(f"AI: {response}\n")

if __name__ == "__main__":
    main()

# PS C:\Users\hi\Desktop\projects\python_projects\ai_projects\LLAMA\TODO_ollama_basics> & C:/Users/hi/Desktop/projects/python_projects/ai_projects/LLAMA/TODO_ollama_basics/.venv/Scripts/python.exe c:/Users/hi/Desktop/projects/python_projects/ai_projects/LLAMA/TODO_ollama_basics/talk_to_llama3_via_API.py

# Welcome to the Local LLM Console Application (Powered by Ollama)
# Type 'exit' to quit the application.

# You: hello
# AI: Hello! It's nice to meet you. Is there something I can help you with or would you like to chat?

# You: exit
# Goodbye!
