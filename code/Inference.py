import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Constants
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# Define the model architecture
def Net():
    """
    Creates a modified ResNet50 model for classification with 133 output classes.
    """
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False   

    # Replace the fully connected layer
    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_fn(model_dir):
    """
    Loads the trained model from the specified directory and prepares it for inference.
    """
    logger.info(f"In model_fn. Model directory is: {model_dir}")
    
    # Initialize the model
    model = Net().to(device)
    
    # Load the model checkpoint
    model_path = os.path.join(model_dir, "model.pth")
    logger.info(f"Loading the dog-classifier model from: {model_path}")
    try:
        with open(model_path, "rb") as f:
            checkpoint = torch.load(f, map_location=device)
            logger.info(f"Checkpoint keys: {checkpoint.keys()}")  # Log checkpoint keys
            model.load_state_dict(checkpoint)
            logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    # Set the model to evaluation mode
    model.eval()
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    """
    Processes input data and converts it into a format suitable for the model.
    """
    logger.info(f"Deserializing the input data. Content type: {content_type}")
    
    try:
        # Process an image uploaded to the endpoint
        if content_type == JPEG_CONTENT_TYPE:
            logger.info("Processing image from request body.")
            image = Image.open(io.BytesIO(request_body))
            logger.info(f"Image size: {image.size}")  # Log image size
            return image
        
        # Process a URL submitted to the endpoint
        if content_type == JSON_CONTENT_TYPE:
            logger.info("Processing image from URL.")
            request = json.loads(request_body)
            url = request['url']
            logger.info(f"Fetching image from URL: {url}")
            img_content = requests.get(url).content
            image = Image.open(io.BytesIO(img_content))
            logger.info(f"Image size: {image.size}")  # Log image size
            return image
        
        raise Exception(f"Unsupported content type: {content_type}")
    
    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")
        raise

def predict_fn(input_object, model):
    """
    Generates predictions for the given input using the specified model.
    """
    logger.info("In predict_fn")
    
    # Define the transformation pipeline
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Apply transformations to the input image
    logger.info("Transforming input image.")
    input_object = test_transform(input_object)
    logger.info(f"Transformed input shape: {input_object.shape}")  # Log input shape
    
    # Move input tensor to the same device as the model
    input_object = input_object.to(device)
    
    # Perform inference
    with torch.no_grad():
        logger.info("Calling model for prediction.")
        prediction = model(input_object.unsqueeze(0))
        logger.info(f"Prediction shape: {prediction.shape}")  # Log prediction shape
    
    return prediction