from dotenv import load_dotenv
load_dotenv()

#NOTE: Here we provide the context and the returned response is what we wanted
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain

docA = Document(
    page_content="LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production)."
)

model = ChatOpenAI(
    temperature=0.4,
    model='gpt-3.5-turbo-1106'
)

prompt = ChatPromptTemplate.from_template("""
Answer the user's question.
Context: {doc_context}
Question: {input}
""")

chain = prompt | model

response = chain.invoke({
    "input": "What is LCEL?",
    "doc_context": [docA, ], # an array of multiple documents
})

print(response)
"""
content='LCEL stands for LangChain Expression Language, which is a declarative way to easily compose chains together. It was designed to support putting prototypes into production with no code changes, from simple chains to complex ones with hundreds of steps.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 105, 'total_tokens': 151, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-1106', 'system_fingerprint': 'fp_c0b1db30f1', 'finish_reason': 'stop', 'logprobs': None} id='run-b51d5e69-5291-45e6-9c51-f6055bfbd288-0' usage_metadata={'input_tokens': 105, 'output_tokens': 46, 'total_tokens': 151, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}
"""