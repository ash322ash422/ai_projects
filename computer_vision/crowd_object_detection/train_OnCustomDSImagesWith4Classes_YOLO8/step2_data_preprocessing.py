import os, shutil, random
import json
from pathlib import Path

# Formatting bounding boxes to YOLO format ensures compatibility with the YOLO training pipeline.
# Normalizing coordinates (converting to values between 0 and 1) aligns with model input requirements.
# Maintaining or adjusting class IDs ensures the data matches the model's label expectations.

# Following target directives hold the processed data that will be in YOLO8 format
TRAIN_ANN_SOURCE_DIR_STEP1    = Path.cwd() / "temp" / "data" / "train" / "ann"  # Directory containing source JSON files
TRAIN_LABELS_TARGET_DIR_STEP1 = Path.cwd() / "temp" / "data_preprocessed_step1" / "train" /  "labels"  # Directory to save YOLO annotation files
TEST_ANN_SOURCE_DIR_STEP1     = Path.cwd() / "temp" / "data" / "test" / "ann"  # Directory containing source JSON files
TEST_LABELS_TARGET_DIR_STEP1  = Path.cwd() / "temp" / "data_preprocessed_step1" / "test" /  "labels"  # Directory to save YOLO annotation files

TRAIN_IMG_SOURCE_DIR_STEP1    = Path.cwd() / "temp" / "data" / "train" / "img"  
TRAIN_IMAGES_TARGET_DIR_STEP1 = Path.cwd() / "temp" / "data_preprocessed_step1" / "train" /  "images" 
TEST_IMG_SOURCE_DIR_STEP1     = Path.cwd() / "temp" / "data" / "test" / "img"  # Directory containing source img files
TEST_IMAGES_TARGET_DIR_STEP1  = Path.cwd() / "temp" / "data_preprocessed_step1" / "test" /  "images"  


#Following target directories hold random files selected from the previous step.
TRAIN_LABELS_SOURCE_DIR_STEP2 = TRAIN_LABELS_TARGET_DIR_STEP1
TRAIN_LABELS_TARGET_DIR_STEP2 = Path.cwd() / "temp" / "data_preprocessed_step2" / "train" /  "labels"  # Directory to save YOLO annotation files
VALID_LABELS_SOURCE_DIR_STEP2 = TRAIN_LABELS_TARGET_DIR_STEP1
VALID_LABELS_TARGET_DIR_STEP2 = Path.cwd() / "temp" / "data_preprocessed_step2" / "valid" /  "labels"  # Directory to save YOLO annotation files
TEST_LABELS_SOURCE_DIR_STEP2  = TEST_LABELS_TARGET_DIR_STEP1
TEST_LABELS_TARGET_DIR_STEP2  = Path.cwd() / "temp" / "data_preprocessed_step2" / "test" /  "labels"  # Directory to save YOLO annotation files

TRAIN_IMG_SOURCE_DIR_STEP2    = TRAIN_IMAGES_TARGET_DIR_STEP1
TRAIN_IMAGES_TARGET_DIR_STEP2 = Path.cwd() / "temp" / "data_preprocessed_step2" / "train" /  "images" 
VALID_IMG_SOURCE_DIR_STEP2    = TRAIN_IMAGES_TARGET_DIR_STEP1
VALID_IMAGES_TARGET_DIR_STEP2 = Path.cwd() / "temp" / "data_preprocessed_step2" / "valid" /  "images"  # Directory to save YOLO annotation files
TEST_IMG_SOURCE_DIR_STEP2     = TEST_IMAGES_TARGET_DIR_STEP1
TEST_IMAGES_TARGET_DIR_STEP2  = Path.cwd() / "temp" / "data_preprocessed_step2" / "test" /  "images"  

CLASS_MAPPING = {                   # Map classTitle to numeric IDs
    "curved mayo scissor": 0,
    "straight mayo scissor": 1,
    "scalpel n\u00ba4": 2,
    "straight dissection clamp": 3,  
}

NUM_FILES_TRAIN_VALID = 400 # total size for training and validation dataset #TODO run1(300) and run2(400)
PERCENT_TRAIN = 0.8 # percentage for training and rest for validation dataset
NUM_FILES_TEST = int(0.1 * NUM_FILES_TRAIN_VALID) # grab just fraction for testing or any number that you want to
        
def convert_to_yolo_format(source_file, target_dir, class_mapping):
    # Load the source dataset
    with open(source_file, 'r') as f:
        data = json.load(f)

    # Image size
    img_width = data["size"]["width"]
    img_height = data["size"]["height"]

    # Initialize the YOLO annotation content
    yolo_annotations = []

    # Process each object in the dataset
    for obj in data["objects"]:
        # Get classTitle and its corresponding numeric ID
        class_title = obj["classTitle"]
        if class_title not in class_mapping:
            print(f"Skipping unknown classTitle: {class_title}")
            continue
        class_id = class_mapping[class_title]

        # Get bounding box coordinates
        x_min, y_min = obj["points"]["exterior"][0]
        x_max, y_max = obj["points"]["exterior"][1]

        # Normalize coordinates
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # Append annotation in YOLO format
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save annotations to a YOLO-format text file
    base_filename = os.path.splitext(os.path.basename(source_file))[0][:-4] # remove '.jpg'
    yolo_file_path = os.path.join(target_dir, f"{base_filename}.txt")
    with open(yolo_file_path, 'w') as f:
        f.write("\n".join(yolo_annotations))

    print(f"Annotations saved to {yolo_file_path}")
    
def reformat_directory_files_to_YOLO8(source_dir, target_dir):
    """Process all JSON files in the source directory."""
    for filename in os.listdir(source_dir):
        if filename.endswith(".json"):
            source_file_path = os.path.join(source_dir, filename)
            convert_to_yolo_format(source_file_path, target_dir,CLASS_MAPPING)

def copy_files(source_dir, target_dir):
    """Copy all files from the source directory to the target directory."""
    # Iterate over all files in the source directory
    for file_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file_name)
        
        # Check if it is a file (skip subdirectories)
        if os.path.isfile(source_path):
            target_path = os.path.join(target_dir, file_name)
            
            # Move the file
            shutil.copy(source_path, target_path)
            print(f"Copied: {file_name}")

    print(f"All files moved from {source_dir} to {target_dir}")

def split_and_copy_train_valid_files(images_dir, labels_dir, 
                         target_train_images_dir, target_train_labels_dir, 
                         target_valid_images_dir, target_valid_labels_dir, 
                         num_files, percent_train):
    """Split files into training and validation sets and copy them with their labels."""
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    
    # Check if there are enough files
    if len(image_files) < num_files:
        print(f"Warning: Only {len(image_files)} image files available.")
        num_files = len(image_files)

    # Randomly shuffle and split files into 80% training and 20% validation
    random.shuffle(image_files)
    split_index = int(percent_train * num_files)
    train_files = image_files[:split_index]
    valid_files = image_files[split_index:num_files]

    # Ensure target directories exist
    os.makedirs(target_train_images_dir, exist_ok=True)
    os.makedirs(target_train_labels_dir, exist_ok=True)
    os.makedirs(target_valid_images_dir, exist_ok=True)
    os.makedirs(target_valid_labels_dir, exist_ok=True)

    # Copy training files and their labels
    for image_file in train_files:
        # Source and target paths for training images
        image_source_path = os.path.join(images_dir, image_file)
        image_target_path = os.path.join(target_train_images_dir, image_file)

        # Source and target paths for training labels
        label_file = f"{image_file[:-4]}.txt"  # Corresponding label file
        label_source_path = os.path.join(labels_dir, label_file)
        label_target_path = os.path.join(target_train_labels_dir, label_file)

        # Check if label exists
        if not os.path.exists(label_source_path):
            print(f"Warning: Label file for {image_file} not found, skipping.")
            continue

        # Copy image and label files
        shutil.copy(image_source_path, image_target_path)
        shutil.copy(label_source_path, label_target_path)
        print(f"Copied to training set: {image_file} and {label_file}")

    # Copy validation files and their labels
    for image_file in valid_files:
        # Source and target paths for validation images
        image_source_path = os.path.join(images_dir, image_file)
        image_target_path = os.path.join(target_valid_images_dir, image_file)

        # Source and target paths for validation labels
        label_file = f"{image_file[:-4]}.txt"  # Corresponding label file
        label_source_path = os.path.join(labels_dir, label_file)
        label_target_path = os.path.join(target_valid_labels_dir, label_file)

        # Check if label exists
        if not os.path.exists(label_source_path):
            print(f"Warning: Label file for {image_file} not found, skipping.")
            continue

        # Copy image and label files
        shutil.copy(image_source_path, image_target_path)
        shutil.copy(label_source_path, label_target_path)
        print(f"Copied to validation set: {image_file} and {label_file}")

    print(f"Split {num_files} files into 80% training and 20% validation sets.")

def copy_random_test_files(images_dir, labels_dir, target_images_dir, target_labels_dir, num_files):
    """Randomly select files and copy them along with their corresponding labels."""
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    
    # Check if there are enough files
    if len(image_files) < num_files:
        print(f"Warning: Only {len(image_files)} image files available.")
        num_files = len(image_files)

    # Randomly select the specified number of image files
    selected_files = random.sample(image_files, num_files)

    # Move the selected files and their corresponding labels
    for image_file in selected_files:
        # Image source and target paths
        image_source_path = os.path.join(images_dir, image_file)
        image_target_path = os.path.join(target_images_dir, image_file)

        # Label source and target paths
        label_file = f"{image_file[:-4]}.txt"  # Corresponding label file , remove last 4 char '.jpg'
        label_source_path = os.path.join(labels_dir, label_file)
        label_target_path = os.path.join(target_labels_dir, label_file)

        # Check if the label file exists
        if not os.path.exists(label_source_path):
            print(f"Warning: Label file for {image_file} not found, skipping.")
            continue

        # Move image and label files
        shutil.copy(image_source_path, image_target_path)
        shutil.copy(label_source_path, label_target_path)
        print(f"Copied: {image_file} and {label_file}")

    print(f"Randomly selected {num_files} image files and copied them along with their labels.")


def main():
    #STEP1: convert it into YOLO8 format and move the data into different directories
    shutil.rmtree(TRAIN_LABELS_TARGET_DIR_STEP1) if os.path.exists(TRAIN_LABELS_TARGET_DIR_STEP1) else None
    os.makedirs(TRAIN_LABELS_TARGET_DIR_STEP1, exist_ok=True)
    reformat_directory_files_to_YOLO8(TRAIN_ANN_SOURCE_DIR_STEP1, TRAIN_LABELS_TARGET_DIR_STEP1)
    
    shutil.rmtree(TRAIN_IMAGES_TARGET_DIR_STEP1) if os.path.exists(TRAIN_IMAGES_TARGET_DIR_STEP1) else None
    os.makedirs(TRAIN_IMAGES_TARGET_DIR_STEP1, exist_ok=True)
    copy_files(TRAIN_IMG_SOURCE_DIR_STEP1,TRAIN_IMAGES_TARGET_DIR_STEP1)

    shutil.rmtree(TEST_LABELS_TARGET_DIR_STEP1) if os.path.exists(TEST_LABELS_TARGET_DIR_STEP1) else None
    os.makedirs(TEST_LABELS_TARGET_DIR_STEP1, exist_ok=True)
    reformat_directory_files_to_YOLO8(TEST_ANN_SOURCE_DIR_STEP1, TEST_LABELS_TARGET_DIR_STEP1)
    
    shutil.rmtree(TEST_IMAGES_TARGET_DIR_STEP1) if os.path.exists(TEST_IMAGES_TARGET_DIR_STEP1) else None
    os.makedirs(TEST_IMAGES_TARGET_DIR_STEP1, exist_ok=True)
    copy_files(TEST_IMG_SOURCE_DIR_STEP1,TEST_IMAGES_TARGET_DIR_STEP1)
        
    
    #STEP2: Randomly select NUM_FILES_TRAIN_VALID datasets  and move it into different directory
    shutil.rmtree(TRAIN_IMAGES_TARGET_DIR_STEP2) if os.path.exists(TRAIN_IMAGES_TARGET_DIR_STEP2) else None
    os.makedirs(TRAIN_IMAGES_TARGET_DIR_STEP2, exist_ok=True) # Ensure target directories exist
    
    shutil.rmtree(TRAIN_LABELS_TARGET_DIR_STEP2) if os.path.exists(TRAIN_LABELS_TARGET_DIR_STEP2) else None
    os.makedirs(TRAIN_LABELS_TARGET_DIR_STEP2, exist_ok=True) # Ensure target directories exist
    shutil.rmtree(VALID_IMAGES_TARGET_DIR_STEP2) if os.path.exists(VALID_IMAGES_TARGET_DIR_STEP2) else None
    os.makedirs(VALID_IMAGES_TARGET_DIR_STEP2, exist_ok=True) # Ensure target directories exist
    shutil.rmtree(VALID_LABELS_TARGET_DIR_STEP2) if os.path.exists(VALID_LABELS_TARGET_DIR_STEP2) else None
    os.makedirs(VALID_LABELS_TARGET_DIR_STEP2, exist_ok=True) # Ensure target directories exist
    split_and_copy_train_valid_files(TRAIN_IMAGES_TARGET_DIR_STEP1, TRAIN_LABELS_TARGET_DIR_STEP1, 
                        TRAIN_IMAGES_TARGET_DIR_STEP2, TRAIN_LABELS_TARGET_DIR_STEP2,
                        VALID_IMAGES_TARGET_DIR_STEP2, VALID_LABELS_TARGET_DIR_STEP2,
                        num_files=NUM_FILES_TRAIN_VALID, percent_train=PERCENT_TRAIN
    )
    
    shutil.rmtree(TEST_IMAGES_TARGET_DIR_STEP2) if os.path.exists(TEST_IMAGES_TARGET_DIR_STEP2) else None
    os.makedirs(TEST_IMAGES_TARGET_DIR_STEP2, exist_ok=True) # Ensure target directories exist
    shutil.rmtree(TEST_LABELS_TARGET_DIR_STEP2) if os.path.exists(TEST_LABELS_TARGET_DIR_STEP2) else None
    os.makedirs(TEST_LABELS_TARGET_DIR_STEP2, exist_ok=True) # Ensure target directories exist
    copy_random_test_files(TEST_IMAGES_TARGET_DIR_STEP1, TEST_LABELS_TARGET_DIR_STEP1, 
                    TEST_IMAGES_TARGET_DIR_STEP2, TEST_LABELS_TARGET_DIR_STEP2,
                    num_files = NUM_FILES_TEST
    )
    
if __name__ == '__main__':
    main()