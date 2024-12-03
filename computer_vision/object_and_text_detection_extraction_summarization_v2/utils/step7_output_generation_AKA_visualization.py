import json
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3, os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

base_dir = Path.cwd().parent / "data"
# Input and output file
input_image = base_dir / "input_images" / "input_image.jpg" #input
mapped_data_json_file  = base_dir / "output" / "mapped_data.json"#input
db_name  = base_dir / "output" / "object_metadata.db" #input
annotated_image_file = base_dir / "output" / "annotated_image.jpg" # Output 
def _get_bounding_boxes(db_name):
    # Format: {image_name: (x1, y1, x2, y2)}
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    query = """
    SELECT object_image_path, x1, y1, x2, y2 FROM objects
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    bounding_boxes = {os.path.basename(row[0]): (row[1], row[2], row[3], row[4]) for row in rows}
    conn.close()
    
    return bounding_boxes

# Function to generate final output
def generate_final_output(master_image_path, mapped_data_json_file, db_name, annotated_image_file):
    # Load the master image
    master_image = Image.open(master_image_path).convert("RGB")
    draw = ImageDraw.Draw(master_image)
    
    # Load mapped data from JSON
    with open(mapped_data_json_file, "r") as f:
        mapped_data = json.load(f)
    
    objects = mapped_data["objects"]
    
    # Get bounding box coordinates
    bounding_boxes = _get_bounding_boxes(db_name)
    
    # Annotate image with bounding boxes and object IDs
    for obj in objects:
        image_name = obj["image_name"]
        unique_id = obj["unique_id"]
        bbox = bounding_boxes.get(image_name, None)
        
        if bbox:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), unique_id, fill="red")
    
    # Save annotated image
    master_image.save(annotated_image_file)
    
    # Create a summary table using pandas
    table_data = {
        "Unique ID": [obj["unique_id"] for obj in objects],
        "Image Name": [obj["image_name"] for obj in objects],
        "Description": [obj["object_description"] for obj in objects],
        "Extracted Text": [obj["extracted_text"] for obj in objects],
        "Summary": [obj["summary"] for obj in objects],
    }
    df = pd.DataFrame(table_data)
    
    # Display the final visual output
    fig, ax = plt.subplots(2, 1, figsize=(10, 15))  # 2 rows, 1 column
    
    # Show annotated image
    ax[0].imshow(master_image)
    ax[0].axis("off")
    ax[0].set_title("Annotated Master Image", fontsize=14)
    
    # Show table in the second subplot
    ax[1].axis("tight")
    ax[1].axis("off")
    ax[1].set_title("Mapped Data Summary", fontsize=14)
    
    # Create a table with adjusted font size
    table = ax[1].table(
        cellText=df.values.tolist(),
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5)  # Adjust font size as needed
    table.auto_set_column_width(col=list(range(len(df.columns))))  # Adjust column widths
    
    plt.tight_layout()
    plt.show()
    
    return annotated_image_file
    
def main():
    _out_file1 = generate_final_output(
        master_image_path=input_image,#input
        mapped_data_json_file=mapped_data_json_file,#input
        db_name=db_name,#input
        annotated_image_file=annotated_image_file, #output
    )
    print(f"Annotated image saved to {_out_file1}")
    
if __name__ == "__main__":
    main()

