import os
import pandas as pd
import easyocr
from pathlib import Path

base_dir = Path.cwd().parent / "data"
extracted_objects_dir = base_dir / "output" / "extracted_objects" #input
extracted_text_file = base_dir / "output" / "extracted_text.csv" #output
       
# Function to extract text from images
def extract_text_from_images(extracted_objects_dir, extracted_text_file):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # Specify the languages, e.g., English ('en')

    extracted_text = []
    
    for image_file in sorted(os.listdir(extracted_objects_dir)):
        image_path = os.path.join(extracted_objects_dir, image_file)
        try:
            # Perform OCR on the image. detail=0 for plain text output
            results = reader.readtext(image_path, detail=0)  
            
            # Combine all detected text into a single string
            extracted_text = " ".join(results) if results else "No text detected"
            
            # Store image name and extracted text
            extracted_text.append((image_file, extracted_text))
        except Exception as e:
            extracted_text.append((image_file, f"Error: {e}"))
            
    # Save extracted text to a CSV file
    df = pd.DataFrame(extracted_text, columns=["Image Name", "Extracted Text"])
    df.to_csv(extracted_text_file, index=False) 

    return extracted_text_file, extracted_text

def main():
    _out_file,_ = extract_text_from_images(
        extracted_objects_dir, #input
        extracted_text_file #output
    )
    
    print(f"--Extracted text data saved in {_out_file}")
    
if __name__ == '__main__':
    main()
