# app.py
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from langchain_community.chat_models import ChatOpenAI  # Chat model initialization
from langchain_experimental.agents import create_pandas_dataframe_agent

#NOTE: this lack ability to create visual plot
# Initialize the LangChain LLM (replace 'your-api-key' with your actual API key)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

st.title("LLM-Powered Data Analysis and Visualization")
st.write("Upload your structured data (CSV or Excel) and ask questions.")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

if uploaded_file:
    # Step 2: Parse and Preview Data
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()
        
        st.write("### Data Preview:")
        st.dataframe(df.head())

        # Step 3: User Query
        user_query = st.text_input("Ask a question about your data:")
        
        if user_query:
            # Step 4: Use LangChain Agent
            # Enable execution of dangerous code (opt-in required)
            agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
            
            try:
                # Run the query
                with st.spinner("Analyzing your data..."):
                    result = agent.invoke(user_query)
                
                # Display the result
                st.write("### Result:")
                st.write(result)
                
            except Exception as e:
                st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Failed to process the file. Error: {e}")