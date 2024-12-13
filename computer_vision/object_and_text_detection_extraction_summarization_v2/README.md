# Instructions
I used python3.11

The input_image is located in data/input_images/ directory. All the output is directed to data/output/ directory.

You can run this program through command line or streamlit GUI. The command line would create one additional image during execution of step7_output_generation_AKA_visualization.py which contains tables.

The file names are verbose. I have chosen to make the filename as descriptive as possible. Modify them as per your needs.

To pass data from one method to another, I have made use of global variables in the form of files,directories and database.

# Command line
Execute each file sequentially: models/{step1_segmentation_model.py, step2_object_extraction_model.py, step3_object_identification_model.py, step4_text_extraction_model.py,step5_summarization_model.py} and then execute utils/{step6_data_mapping.py, step7_output_generation_AKA_visualization.py}.

# Streamlit GUI
Load the input_image located in data/input_images/input_image.jpg

