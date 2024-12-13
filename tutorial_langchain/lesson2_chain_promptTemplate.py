from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Instantiate Model
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-3.5-turbo-1106",
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma separated list."),
        ("human", "{input}")
    ]
)

# Create LLM Chain
chain = prompt | llm

response = chain.invoke({"input": "happy"})
print(response)

"""
content='joyful, content, delighted, pleased, satisfied, cheerful, elated, ecstatic, glad, jubilant' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 33, 'total_tokens': 56, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-1106', 'system_fingerprint': 'fp_c0b1db30f1', 'finish_reason': 'stop', 'logprobs': None} id='run-044579e5-ad7f-41d7-829a-6817aee67524-0' usage_metadata={'input_tokens': 33, 'output_tokens': 23, 'total_tokens': 56, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}
"""