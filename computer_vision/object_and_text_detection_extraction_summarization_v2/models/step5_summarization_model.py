import pandas as pd
from transformers import pipeline
from pathlib import Path

base_dir = Path.cwd().parent / "data"
extracted_text_file  = base_dir / "output" / "extracted_text.csv" #input
summarized_object_attributes_file = base_dir / "output" / "summarized_object_attributes.csv"#output

# Function to summarize object attributes
def summarize_object_attributes(extracted_text_file, summarized_object_attributes_file):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Load the extracted text data
    df = pd.read_csv(extracted_text_file)
    
    summaries = []
    for index, row in df.iterrows():
        image_name = row['Image Name']
        text = row['Extracted Text']
        
        try:
            # Skip summarization if no text detected
            if text == "No text detected" or text.startswith("Error"):
                summaries.append((image_name, "Summary not applicable"))
                continue
            
            # Summarize the text
            summary = summarizer(text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
            summaries.append((image_name, summary))
        except Exception as e:
            summaries.append((image_name, f"Error during summarization: {e}"))
    
    # Save summarized attributes to a new CSV file
    summary_df = pd.DataFrame(summaries, columns=["Image Name", "Summary"])
    summary_df.to_csv(summarized_object_attributes_file, index=False)

    return summarized_object_attributes_file,summaries

def main():
    _out_file,_ = summarize_object_attributes(
        extracted_text_file, #input
        summarized_object_attributes_file #output
    )
    print(f"--Summarized attributes saved in {_out_file}")
    
if __name__ == "__main__":
    main()
    
    