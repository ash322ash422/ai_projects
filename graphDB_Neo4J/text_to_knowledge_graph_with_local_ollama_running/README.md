1)I used python3.11.

2) * Install 'ollama' from https://ollama.com/download. 
   * Now we search for models. Goto https://ollama.com/search and find your model. I picked llama3.2
   * Now open window terminal and type 'ollama run llama3.2'. If the model does not exist locally, it will install it, otherwise prompt appears. it took around 3 min. for it to install
   * If you goto http://localhost:11434/ , you should see message "Ollama is running".


3) I also installed following packages using command:
  jupyter==1.1.1 , langchain==0.3.8 , langchain-community==0.3.8 , langchain-groq==0.2.1, neo4j==5.26.0 , pypdf==5.1.0 , wikipedia==1.4.0 , youtube-transcript-api==0.6.3 , tiktoken==0.8.0 , pandas==2.2.3 , python-dotenv==1.0.1 , transformers==4.46.3
 
 * python -m pip install jupyter langchain langchain-community langchain-groq neo4j pypdf wikipedia youtube-transcript-api tiktoken pandas python-dotenv transformers
  
4) Now in project dir, type 'python -m jupyter notebook'
 
 