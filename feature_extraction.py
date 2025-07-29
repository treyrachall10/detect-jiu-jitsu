import os
import csv
import json
import torch
from PIL import Image
import pandas as pd
import consts
from utils import extract_number
from torchvision.transforms import v2

# Converts the frame to a tensor of shape [1, 3, 256, 128]
def image_to_tensor(image, image_size):
    """
    Crops a region from the specified frame and converts it into a normalized tensor.

    The output tensor has shape [3, 256, 128] and is normalized using ImageNet statistics.
    This format is typically used for appearance-based feature extraction models.

    Parameters:
        frame (int): The frame number to retrieve and crop from the video.
        x1 (int): Top-left x-coordinate of the bounding box in pixels.
        y1 (int): Top-left y-coordinate of the bounding box in pixels.
        x2 (int): Bottom-right x-coordinate of the bounding box in pixels.
        y2 (int): Bottom-right y-coordinate of the bounding box in pixels.

    Returns:
        torch.Tensor: A 3-channel tensor of shape [3, 256, 128] ready for model input.
    """
    # Resize image to shape (256, 128) and convert to normalized tensor
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    return transforms(image)

def extract_and_process_image(model_type, image_size):
    """
    Extracts features from manually cropped images using an osnet model.

    Images are loaded, sorted numerically, batched, and passed through the model.
    Extracted features are saved to a CSV file for later use.

    Returns:
        None
    """
    print('Extracting features...')
    # Creates a numerically sorted list of file names 
    image_names = sorted(os.listdir(f'crops/{consts.video_name}'), key=extract_number)

    # Create and open a file to write filenames and their corresponding extracted features
    with open(f"csv_output/has_features/{consts.features_file_name}", mode="w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'features'])
        writer.writeheader()

        batch_size = 64
        current_batch = []

        # Iterate through every image and convert to tensor
        for i, filename in enumerate(image_names):
            path = f"crops/{consts.video_name}/{filename}"
            try:
                img = Image.open(path)
                tensor = image_to_tensor(img, image_size=image_size)
                current_batch.append((filename, tensor))
                img.close()
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
            # If batch is full or on last image extract features and write to file
            if len(current_batch) == batch_size or i == len(image_names) - 1:
                batch_tensor = torch.stack([t[1] for t in current_batch])

                with torch.no_grad():
                    print("Input to extractor shape:", batch_tensor.shape)
                    features = consts.extractor(batch_tensor, label=None, cam_label=None, view_label=None)

                for (filename, _), feature in zip(current_batch, features):
                    writer.writerow({
                        'filename': filename,
                        'features': json.dumps(feature.tolist())
                    })
                current_batch.clear()
    
    print('Finished getting features')

def transformer_extractor():
    """
    Extracts features from manually cropped images using a transformer model.

    Images are loaded, sorted numerically, batched, and passed through the model.
    Extracted features are saved to a CSV file for later use.

    Returns:
        None
    """
    print('Extracting features...')
    # Creates a numerically sorted list of file names 
    image_names = sorted(os.listdir(f'crops/{consts.video_name}'), key=extract_number)

    # Create and open a file to write filenames and their corresponding extracted features
    with open(f"csv_output/has_features/{consts.features_file_name}", mode="w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'features'])
        writer.writeheader()

        batch_size = 64
        img_batch = []
        name_batch = []

        # Iterate through every image and convert to tensor
        for i, filename in enumerate(image_names):
            path = f"crops/{consts.video_name}/{filename}"
            try:
                img = Image.open(path)
                img_batch.append(img)
                name_batch.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
            # If batch is full or on last image extract features and write to file
            if len(img_batch) == batch_size or i == len(image_names) - 1:

                with torch.no_grad():
                    inputs = consts.processor(images=img_batch, return_tensors="pt")
                    output = consts.extractor(**inputs)
                    features = output.last_hidden_state[:, 0, :]

                for img in img_batch:
                    img.close()

                for name, feature in zip(name_batch, features):
                    writer.writerow({
                        'filename': name,
                        'features': json.dumps(feature.tolist())
                    })
                img_batch.clear()
                name_batch.clear()
    
    print('Finished getting features')

def extract_features():
    if 'osnet' in consts.extractor_name:
        extract_and_process_image(image_size=(256, 128))
    elif 'transreid' in consts.extractor_name:
        extract_and_process_image(image_size=(384, 128))
    elif 'transformer' in consts.extractor_name:
        transformer_extractor()
