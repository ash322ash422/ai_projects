import os
import json
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import sys
# Add directories to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

# Ensure directories exist and are writable

from conf import EXTRACTED_SEG_REGIONS
from pathlib import Path

base_dir = Path.cwd() / "data"
input_images_dir = base_dir / "input_images" 
image_path = base_dir / "input_images" / "input_image.jpg"
segmented_file = base_dir / "output" / "segmented_output.png"
extracted_seg_regions = base_dir / "output" / EXTRACTED_SEG_REGIONS
extracted_objects_dir = base_dir / "output" / "extracted_objects"
db_name = base_dir / "output" / "object_metadata.db"
object_descriptions_file = base_dir / "output" / "object_descriptions.csv"
extracted_text_file = base_dir / "output" / "extracted_text.csv" 
summarized_object_attributes_file = base_dir / "output" / "summarized_object_attributes.csv"
mapped_data_json_file = base_dir / "output" / "mapped_data.json"  # JSON file from Step 6
annotated_image_file = base_dir / "output" / "annotated_master_image.jpg"  #  annotated image

# Import functions from models and utils
from step1_segmentation_model import segment_image
from step2_object_extraction_model import extract_objects
from step3_object_identification_model import identify_objects
from step4_text_extraction_model import extract_text_from_images
from step5_summarization_model import summarize_object_attributes
from step6_data_mapping import map_data_to_json
from step7_output_generation_AKA_visualization import generate_final_output

# def preprocess_image(image):
#     """ Example preprocessing step: resize image """
#     resized_image = cv2.resize(image, (256, 256))
#     return resized_image

def save_uploaded_file(uploaded_file, save_path):
    """ Save the uploaded file to the specified path """
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

def main():
    st.title("AI Pipeline for Image Segmentation and Object Analysis")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        # st.write(Path.cwd())
        # Save uploaded file to the input images directory
        image_path = Path.cwd() /  "data" / "input_images" / uploaded_file.name
        save_uploaded_file(uploaded_file, image_path)
        
        # Load and display the uploaded image
        image = Image.open(image_path)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        # image_cv = cv2.imread(image_path)
        # preprocessed_image = preprocess_image(image_cv)
        # preprocessed_image_path = os.path.join(output_dir, f'preprocessed_{uploaded_file.name}')
        # cv2.imwrite(preprocessed_image_path, preprocessed_image)
        
        st.write("Running segmentation...")
        segmented_filename, extracted_seg_regions_dir = segment_image(
        image_path=image_path,#input
        segmented_file=segmented_file, #output
        extracted_seg_regions=extracted_seg_regions #output
        )
        image_seg = Image.open(segmented_file)
        st.image(image_seg, caption='Segmented Image', use_column_width=True)
        st.write(f"...segmented_filename : {segmented_filename}")
        st.write(f"...extracted_seg_regions_dir : {extracted_seg_regions_dir}")
        
        st.write("Extracting objects...")
        extracted_objects_directory,db = extract_objects(
            original_image_path=image_path,#input
            mask_dir=extracted_seg_regions,#input
            extracted_objects_dir=extracted_objects_dir,#output
            db_name=db_name#output
        )
        st.write(f"...objects extracted to {extracted_objects_directory}...")
        st.write(f"...db created in {db}...")
        
        st.write("Identifying objects...")
        _out_file,object_descriptions = identify_objects(
            extracted_objects_directory=extracted_objects_directory, #input
            object_descriptions_file=object_descriptions_file #output
        )
        st.write(f"...Identified objects saved in {object_descriptions_file}")
         
        st.write("Extracting text...")
        _out_file,extracted_text = extract_text_from_images(
            extracted_objects_dir=extracted_objects_dir, #input
            extracted_text_file=extracted_text_file #output
        )
        st.write(f"...Extracted text saved in {extracted_text_file}")
        
        st.write("Summarizing attributes...")
        _out_file, summarized_object_attributes = summarize_object_attributes(
            extracted_text_file=extracted_text_file, #input
            summarized_object_attributes_file=summarized_object_attributes_file #output
        )
        st.write(f"...Summarizing attributes saved in {summarized_object_attributes_file}")
        
        st.write("Mapping data...")
        _out_file, mapped_data = map_data_to_json(
            master_image=image_path,#input
            extracted_text_file=extracted_text_file, #input
            summarized_object_attributes_file=summarized_object_attributes_file, #input
            object_description_file=object_descriptions_file,#input
            mapped_data_json_file=mapped_data_json_file # output
        )
        st.write(f"...Mapped data saved in {mapped_data_json_file}")
        # mapped_data_path = os.path.join(output_dir, 'mapped_data.json')
        # with open(mapped_data_path, 'w') as f:
        #     json.dump(mapped_data, f, indent=4)

        st.write("Generating output...")
        _out_file = generate_final_output(
            master_image_path=image_path,#input
            mapped_data_json_file=mapped_data_json_file,#input
            db_name=db_name,#input
            annotated_image_file=annotated_image_file, #output
        )
    
        # final_image_path = os.path.join(output_dir, 'final_image.png')
        # generate_output(image_path, segmented_path, mapped_data_path)
        final_image = Image.open(annotated_image_file)
        st.image(final_image, caption='Processed Image.', use_column_width=True)

        st.write("Pipeline completed.")

if __name__ == "__main__":
    main()
