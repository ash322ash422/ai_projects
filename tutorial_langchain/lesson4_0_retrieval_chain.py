from dotenv import load_dotenv
load_dotenv()

#NOTE: This program  provides no context and the returned response is not what we wanted 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(
    temperature=0.4,
    model='gpt-3.5-turbo-1106'
)

prompt = ChatPromptTemplate.from_template("""
Answer the user's question.
Question: {input}
""")

chain = prompt | model

response = chain.invoke({
    "input": "What is LCEL?",
})

print(response)
"""
content='LCEL stands for "Lowest Common Endangered Language," which is a project that aims to identify and document endangered languages around the world. The goal is to raise awareness about these languages and support efforts to preserve and revitalize them.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 47, 'prompt_tokens': 21, 'total_tokens': 68, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-1106', 'system_fingerprint': 'fp_c0b1db30f1', 'finish_reason': 'stop', 'logprobs': None} id='run-9953aef6-f321-4a02-a0c9-5d3ef66ed52a-0' usage_metadata={'input_tokens': 21, 'output_tokens': 47, 'total_tokens': 68, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}
"""