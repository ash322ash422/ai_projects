
.. image:: thumbnail.png

The repository contains the code supporting a youtube video `<https://youtu.be/IxEpU_R3LFw>`_.

Prequisits
----------
The code is dependent on packages listed in requirements.txt. 

Contents
--------
The repository contains code to connect to ChatGPT & Whisper APIs, and a simple flask app to demonstrate the APIs.
Resulting program provides a web interface which allows user to enter a (verbal) message (question to ChatGPT) which will get transcribed by Whisper via API and then will be redirected to ChatGPT as a question. The response will be returned to the web-app frontend. 

