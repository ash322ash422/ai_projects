
*) Need an open ai API KEY. Without this it is not going to work.

Resulting program provides a web interface which allows user to enter a (verbal) message (question to ChatGPT) 
which will get transcribed by Whisper via API and then will be redirected to ChatGPT as a question. The response
will be returned to the web-app frontend. 

1) Create a openAI API key

create virtual env with python3.12:
2) pip install urllib3 --upgrade
3) pip install openai==0.28
4) pip install gradio

################
- 'python interface_v1.py ' would run a simple gradio app in browser