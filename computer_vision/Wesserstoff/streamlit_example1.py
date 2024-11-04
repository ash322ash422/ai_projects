import streamlit as st
from PIL import Image
import numpy as np
import torch
#, this is used to .
from transformers import pipeline #From the Hugging Face library, load pre-trained model (summarization) for various NLP tasks

# Assuming you have pre-defined detect_objects() and summarize_object() functions
def detect_objects(image):
    # Simulating object detection. Replace with actual detection model.
    detected_objects = [
        {'class': 1, 'name': 'car', 'xmin': 30, 'ymin': 50, 'xmax': 200, 'ymax': 180},
        {'class': 2, 'name': 'person', 'xmin': 220, 'ymin': 60, 'xmax': 300, 'ymax': 180}
    ]
    return detected_objects

def summarize_object(obj, summarizer):
    description = f"Object: This is a {obj['name']} located at ({obj['xmin']},{obj['ymin']}) with size {obj['xmax'] - obj['xmin']}x{obj['ymax'] - obj['ymin']}."
    summary = summarizer(description)[0]['summary_text']
    return summary

# Set up the summarizer model
# This initializes a summarization pipeline from Hugging Face’s transformers library. 
# The pipeline loads a pre-trained model for text summarization, which will be used to summarize 
# the object descriptions created earlier.
summarizer = pipeline('summarization')

# Streamlit App UI
st.title("Object Detection and Summarization Pipeline")

# Step 1: Image Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Step 2: Detect Objects
    st.write("Detecting objects in the image...")
    detected_objects = detect_objects(image)

    # Display detected objects
    st.write(f"Detected {len(detected_objects)} objects in the image.")

    # Step 3: Summarize Objects
    st.write("Summarizing object attributes...")
    summarized_objects = []

    for obj in detected_objects:
        summary = summarize_object(obj, summarizer)
        summarized_objects.append({
            'Object ID': obj['class'],
            'Label': obj['name'],
            'Summary': summary
        })

    # Step 4: Display Summarized Objects
    for obj in summarized_objects:
        st.write(f"Object ID: {obj['Object ID']}")
        st.write(f"Label: {obj['Label']}")
        st.write(f"Summary: {obj['Summary']}")
        st.write("------")

"""
ai_projects\computer_vision\Wesserstoff> streamlit run c:/Users/hi/Desktop/projects/python_projects/ai_projects/computer_vision/Wesserstoff/streamlit_example1.py
"""