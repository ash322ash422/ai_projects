import streamlit as st
import openai
from brain import get_index_for_pdf
import os
# import databutton as db

# Set the title for the Streamlit app
st.title("RAG Enhanced Chatbot")

from dotenv import load_dotenv
load_dotenv()


# Set up the OpenAI API key from databutton secrets
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# os.environ["OPENAI_API_KEY"] = db.secrets.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Define the template for the chatbot prompt
prompt_template = """
    You are a helpful Assistant who answers users' questions based on multiple contexts given to you.
    Keep your answer short and to the point.
    The evidence consists of the context from the PDF extracts with metadata. 
    Carefully focus on the metadata, especially 'filename' and 'page' whenever answering.
    Make sure to add the filename and page number at the end of the sentence you are citing.
    Reply "Not applicable" if the text is irrelevant.
    
    The PDF content is:
    {pdf_extract}
"""


# Use cache_resource instead of cache_data for the vectordb creation
# The @st.cache_resource ensures that if the create_vectordb function is 
# called again with the same set of files and filenames, the cached result 
# will be returned instead of recomputing the vectordb.
@st.cache_resource
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Creating vector database..."):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], #extracts the raw byte content of each file by calling getvalue() on each file in files
            filenames,
            openai.api_key
        )
    return vectordb


def main():
    
    # Upload PDF files using Streamlit's file uploader
    pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    # If PDF files are uploaded, create the vectordb and store it in the session state
    if pdf_files:
        pdf_file_names = [file.name for file in pdf_files]
        st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)


    # Get the current prompt from the session state or set a default value
    prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

    # Display previous chat messages
    for message in prompt:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Get the user's question using Streamlit's chat input
    question = st.chat_input("Ask anything")

    # Handle the user's question
    if question:
        # st.session_state is a dictionary-like object provided by Streamlit to store
        # and persist data across multiple interactions within a user session.
        vectordb = st.session_state.get("vectordb", None)
        if not vectordb:
            st.write("You need to provide a PDF.")
            st.stop()

        # Search the vectordb for similar content to the user's question
        search_results = vectordb.similarity_search(question, k=3)
        for search_result in  search_results:
            print("search_result=",search_result)
        pdf_extract = "\n".join([result.page_content for result in search_results])

        # Update the prompt with the pdf extract
        prompt[0] = {
            "role": "system",
            "content": prompt_template.format(pdf_extract=pdf_extract),
        }

        # Add the user's question to the prompt and display it
        prompt.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Display an empty assistant message while waiting for the response
        with st.chat_message("assistant"):
            botmsg = st.empty()

        # Call ChatGPT with streaming and display the response as it comes
        response = []
        result = ""
        for chunk in openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=prompt, stream=True
        ):
            text = chunk.choices[0].get("delta", {}).get("content")
            if text is not None:
                response.append(text)
                result = "".join(response).strip()
                botmsg.write(result)

        # Add the assistant's response to the prompt
        prompt.append({"role": "assistant", "content": result})

        # Store the updated prompt in the session state
        st.session_state["prompt"] = prompt
        
if __name__ == "__main__":
    main()