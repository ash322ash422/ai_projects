import openai
import config
openai.api_key = config.OPENAI_API_KEY

response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":"You are a helpful assistant."},
            {"role":"user", "content":"Who won  the world series in  2023?"}
            
            
        ]
)
response_message = response['choices'][0]['message']['content']
print("response=",response)