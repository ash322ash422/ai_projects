import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

# Define the image path
IMAGE_PATH = Path.cwd() / 'images' / 'busy_street_scene_with_multiple_objects_like_cars_people_traffic_lights_buildings.jpg'

# Load a pre-trained Mask R-CNN model with a ResNet-50 backbone
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Function to process the image
def process_image(image_path):
    img = Image.open(image_path)  # Open the image using PIL
    transform = T.Compose([T.ToTensor()])  # Transform the image to a tensor
    return transform(img)  # Return the transformed image

# Function to get predictions from the model
def get_prediction(img_path, threshold=0.5):
    img = process_image(img_path)  # Process the image
    with torch.no_grad():  # Disable gradient calculation for faster inference
        prediction = model([img])  # Get the prediction from the model
    pred_score = prediction[0]['scores'].detach().numpy()  # Get prediction scores
    pred_t = [pred_score > threshold]  # Filter predictions based on threshold
    masks = (prediction[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()  # Convert masks to binary format
    return masks  # Return the masks

# Function to plot the masks on the image
def plot_masks(image_path, masks):
    img = Image.open(image_path)  # Open the image using PIL
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))  # Set up the plot with two subplots
    ax[0].imshow(img)  # Show the original image
    ax[0].set_title("Original Image")
    ax[1].imshow(img)  # Show the original image in the second subplot
    for mask in masks:
        ax[1].imshow(mask, alpha=0.5)  # Overlay each mask with some transparency
    ax[1].set_title("Segmented Image")
    plt.axis('off')  # Turn off the axis
    plt.show()  # Display the plot

def main():
    image_path = IMAGE_PATH  # Path to your image
    masks = get_prediction(image_path)  # Get the segmentation masks
    plot_masks(image_path, masks)  # Plot the masks on the image

# Example usage
if __name__ == "__main__":
    main()
