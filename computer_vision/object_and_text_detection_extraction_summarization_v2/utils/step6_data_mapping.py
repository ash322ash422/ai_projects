import pandas as pd
import json
from functools import reduce
from pathlib import Path

base_dir = Path.cwd().parent / "data"

# Input and output files
input_image = base_dir / "input_images" / "input_image.jpg"
object_description_file = base_dir / "output" / "object_descriptions.csv" # Output from Step 3
extracted_text_file = base_dir / "output" / "extracted_text.csv"  # Output from Step 4
summarized_object_attributes_file = base_dir / "output" / "summarized_object_attributes.csv"  # Output from Step 5
mapped_data_json_file = base_dir / "output" / "mapped_data.json"

# Function to map extracted data and summaries
def map_data_to_json(master_image, extracted_text_file, summarized_object_attributes_file,object_description_file, mapped_data_json_file):
    master_image = str(master_image)
    
    object_description_df = pd.read_csv(object_description_file)
    extracted_df  = pd.read_csv(extracted_text_file)
    summarized_df = pd.read_csv(summarized_object_attributes_file)
    
    # Merge data on 'Image Name'
    # merged_df = pd.merge(extracted_df, summarized_df, on="Image Name", how="outer")
    dfs = [extracted_df, summarized_df, object_description_df]
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="Image Name", how="outer"), dfs)

    # Assign unique IDs to each object
    merged_df['Unique ID'] = ["object_" + str(i + 1) for i in range(len(merged_df))]
    
    # Map data to JSON structure
    object_data = []
    for _, row in merged_df.iterrows():
        object_info = {
            "unique_id": row['Unique ID'],
            "image_name": row['Image Name'],
            "object_description": row['Description'],
            "extracted_text": row['Extracted Text'],
            "summary": row['Summary']
        }
        object_data.append(object_info)
    
    # Final JSON structure
    mapped_data = {
        "master_image": master_image,
        "objects": object_data
    }
    
    # Save JSON to file
    with open(mapped_data_json_file, "w") as f:
        json.dump(mapped_data, f, indent=4)
    
    return mapped_data_json_file,mapped_data
    
def main():
    _out_file,_ = map_data_to_json(
        master_image=input_image,#input
        extracted_text_file=extracted_text_file, #input
        summarized_object_attributes_file=summarized_object_attributes_file, #input
        object_description_file=object_description_file,
        mapped_data_json_file=mapped_data_json_file # output
    )
    print(f"--Data mapped and saved to {_out_file}")

if __name__ == "__main__":
    main()
