I am using python3.11

- Cold email generator for services company using groq, langchain and streamlit.
- It allows users to input the URL of a company's careers page. The tool then extracts job listings from 
  that page and generates personalized cold emails. These emails include relevant portfolio links sourced from a vector database, based on the specific job descriptions. 

**Imagine a scenario:**

- Nike needs a Principal Software Engineer and is spending time and resources in the hiring process, on boarding, training etc
- Atliq is Software Development company can provide a dedicated software development engineer to Nike. So, the business development executive (Mohan) from Atliq is going to reach out to Nike via a cold email.
- see imgs/img.png

- Architecture Diagram : imgs/architecture.png

- Set-up
1. Get `GROQ_API_KEY from here: https://console.groq.com/keys. Put this inside `app/.env` (groq is cloud based platform allowing you to run LLAMA3.X fast)
2. - pip install -r requirements.txt
3. - streamlit run app/main.py
   

- Packages installed:
langchain==0.2.14
langchain-community==0.2.12
langchain-groq===0.1.9
unstructured==0.14.6
selenium==4.21.0
chromadb==0.5.0
streamlit==1.35.0
pandas==2.0.2
python-dotenv==1.0.0