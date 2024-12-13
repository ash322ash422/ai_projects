from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # different providers have diff. embeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

# Retrieve Data
def get_docs():
    loader = WebBaseLoader('https://python.langchain.com/docs/expression_language/')
    docs = loader.load()
    # return docs #
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(
        temperature=0.4,
        model='gpt-3.5-turbo-1106'
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question.
    Context: {context}
    Question: {input}
    """)

    # chain = prompt | model
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever() #returns top 5 (?) most relevant documents

    #Following would retrieve the most relevant documents
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


docs = get_docs() #fetch the context data from web,etc
vectorStore = create_vector_store(docs) # store the data in vectorDB
chain = create_chain(vectorStore)

response = chain.invoke({
    "input": "What is LCEL?",
})

print("response=",response)
print("\n\nresponse['answer']=",response['answer'])

"""
SER_AGENT environment variable not set, consider setting it to identify your requests.
response= {'input': 'What is LCEL?', 'context': [Document(metadata={'source': 'https://python.langchain.com/docs/expression_language/', 'title': 'Conceptual guide | 🦜️🔗 LangChain', 'description': 'This section conttains introductions to key parts of LangChain.', 'language': 'en'}, page_content='LangChain Expression Language, or LCEL, is a declarative way to chain LangChain components.'), Document(metadata={'source': 'https://python.langchain.com/docs/expression_language/', 'title': 'Conceptual guide | 🦜️🔗 LangChain', 'description': 'This section contains introductions to key parts of LangChain.', 'language': 'en'}, page_content='LangChaiin Expression Language (LCEL)\u200b'), Document(metadata={'source': 'https://python.langchain.com/docs/expression_language/', 'title': 'Conceptual guide | 🦜️🔗 LangChain', 'description': 'This section contains introdductions to key parts of LangChain.', 'language': 'en'}, page_content='successfully run LCEL chains with 100s of steps in production). To highlight a few of the reasons you might want to use LCEL:'), Document(metadata={'source': 'https://python.langchain.com/docs/expression_language/', 'title': 'Conceptual guide | 🦜️🔗 LangChain', 'description': 'This section contains introductions to key parts of LangChain.', 'language': 'en'},, page_content='With LCEL, all steps are automatically logged to LangSmith for maximum observability and debuggability.')], 'answer': 'LCEL stands for LangChain Expression Language, which is a declarative way to chain LangChain components. It allows you to run chains with hundreds of steps in production and automatically logs all steps to LangSmith for maximum observability and debuggability.'}


response['answer']= LCEL stands for LangChain Expression Language, which is a declarative way to chain LangChain components. It allows you to run chains with hundreds of steps in production and automatically logs all steps to LangSmith for maximum observability and debuggability.
"""