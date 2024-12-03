# app.py
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from langchain_community.chat_models import ChatOpenAI  # Chat model initialization
from langchain_experimental.agents import create_pandas_dataframe_agent
import matplotlib.pyplot as plt  # For rendering plots
import io
import re  # To filter and clean generated code

#TODO I tried to modify app so that it can plot graph based on user input. It did not work. Work on this later. 
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
        
        st.write("### Data Preview for first few rows:")
        st.dataframe(df.head())

        # Step 3: User Query
        user_query = st.text_input("Ask a question about your data or request a visualization (e.g., 'Plot a histogram of column A'):")

        if user_query:
            # Step 4: Use LangChain Agent
            agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

            try:
                # Run the query and intercept the response
                with st.spinner("Analyzing your data and generating a response..."):
                    raw_result = agent.run(user_query)
                    st.write("### Raw result from agent:")
                    st.write(raw_result)

                # Step 5: Extract and Execute Code for Plotting
                if "plot" in user_query.lower() or "visualize" in user_query.lower():
                    # Use refined regex to extract the plotting code
                    code_matches = re.findall(r"Action Input: (df\.[\w]+\(.+?\))", raw_result)
                    if code_matches:
                        for code_snippet in code_matches:
                            try:
                                st.write(f"Executing code:\n{code_snippet}") #debugging
                                
                                # Clean the code snippet by removing any unwanted output
                                # For example, strip anything after a closing parenthesis in plot commands
                                cleaned_code = re.sub(r"\[.*\]", "", code_snippet.strip())  # Remove any extra list-like output

                                # Execute the cleaned-up code snippet
                                exec_globals = {"df": df, "plt": plt, "io": io}
                                exec(cleaned_code, exec_globals)
                                
                                # Check if a plot was created and render it
                                buf = io.BytesIO()
                                plt.savefig(buf, format="png")
                                buf.seek(0)
                                st.image(buf, use_column_width=True)
                                plt.close()
                            except Exception as e:
                                st.error(f"Error executing generated code: {e}")
                                continue
                    else:
                        st.error("Failed to extract valid Python code for plotting.")
                else:
                    # Display non-plot results
                    st.write("### Result:")
                    st.write(raw_result)

            except Exception as e:
                st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Failed to process the file. Error: {e}")