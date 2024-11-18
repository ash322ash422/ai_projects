import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_text_from_pdf(pdf_documents):
    """Get PDF text from the given PDF"""
    text = ""
    for pdf in pdf_documents:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def split_text_into_chunks(text):
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,# number of characters that overlap between adjacent chunks
        length_function=len # fn. to use to determine length of chunks
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore_for_text(text_chunks):
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore): 
    """Conversational agent that can answer user queries in a contextually aware way, using both stored information 
       and recent conversation history
    """
    llm = ChatOpenAI()
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True #Ensures that the conversation history is returned as a series of messages,
                             # enabling continuity in responses based on past context.
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(), #Converts vectorstore into a retriever. The vector store likely
                     # stores preprocessed embeddings of documents or relevant knowledge, allowing the 
                     # retriever to pull the most contextually relevant information based on a user query.
        memory=memory #  Integrates the conversation memory to maintain past interactions, so responses build on previous exchanges
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0: # alternate messages between user and chatbot
            st.write(f"User: {message.content}") 
        else:
            st.write(f"Bot: {message.content}")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs with different genres")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs")
    user_question = st.text_input("Ask questions about PDF documents that you uploaded:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("UPLOAD PDF Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing PDF Documents"):
                raw_text    = get_text_from_pdf(pdf_docs) 
                text_chunks = split_text_into_chunks(raw_text)
                vectorstore = get_vectorstore_for_text(text_chunks)# create vector store
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
