import openai
import gradio as gr
import config
openai.api_key = config.OPENAI_API_KEY

def speech_to_text(audio):
    '''Transcribe audio file to text by whisper.'''
    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe('whisper-1', audio_file)
    return transcript['text']

def greet(audio):
    transcript = speech_to_text(audio)
    return transcript
    
demo = gr.Interface(fn=greet, inputs=gr.Audio(type='filepath'), outputs="text") # should be .wav or mp3
demo.launch()   