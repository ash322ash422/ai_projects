import cv2
import os
import sqlite3
import uuid
from PIL import Image
import numpy as np
import shutil
from pathlib import Path
from conf import EXTRACTED_SEG_REGIONS

base_dir = Path.cwd().parent / "data"
#Input
mask_dir = base_dir / "output" / EXTRACTED_SEG_REGIONS # Directory containing segmented masks
original_image_path = base_dir / "input_images" / "input_image.jpg" # Path to the original input image
# output files
extracted_objects_dir = base_dir / "output" / "extracted_objects"
db_name  = base_dir / "output" / "object_metadata.db"

# Generate a master ID for the original image
def generate_master_id():
    return str(uuid.uuid4())

# Extract segmented objects
def extract_objects(original_image_path,mask_dir,extracted_objects_dir,db_name):
    shutil.rmtree(extracted_objects_dir, ignore_errors=True)
    os.makedirs(extracted_objects_dir)
    
    # Initialize SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create table for storing metadata (with both sets of bounding box coordinates)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS objects (
        id TEXT PRIMARY KEY,
        master_id TEXT,
        object_image_path TEXT,
        x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
    )
    """)
    conn.commit()

    # Load the original image
    original_image = cv2.imread(str(original_image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Master ID for the original image
    master_id = generate_master_id()
    
    # Iterate over segmented masks
    mask_files = sorted(os.listdir(mask_dir))
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Find the bounding box for the object in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate the new bounding box coordinates (x1, y1, x2, y2)
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            
            # Extract the object
            segmented_object = original_image[y:y+h, x:x+w]
            
            # Generate a unique ID for the object
            object_id = str(uuid.uuid4())
            
            # Save the extracted object
            object_filename = str(extracted_objects_dir / f"object_{object_id}.png")
            cv2.imwrite(object_filename, cv2.cvtColor(segmented_object, cv2.COLOR_RGB2BGR))
            
            # Save metadata in the database including both sets of bounding box coordinates
            cursor.execute("""
            INSERT INTO objects (id, master_id, object_image_path,  x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (object_id, master_id, object_filename, x1, y1, x2, y2))
            conn.commit()
            
    # Close database connection
    conn.close()
    print(f"--Extraction complete. Master ID: {master_id}")
    
    return extracted_objects_dir,db_name
def main():
    extracted_objects_directory,db = extract_objects(
        original_image_path=original_image_path,#input
        mask_dir=mask_dir,#input
        extracted_objects_dir=extracted_objects_dir,#output
        db_name=db_name #output
    )
    print(f"--Extracted objects saved in {extracted_objects_directory}")
    print(f"--Object meta-data  saved in {db}")
    
if __name__ == "__main__":
    main()

