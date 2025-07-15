from PIL import Image
import pandas as pd
from torchvision.transforms import v2
import torch
import config
import csv
import json

# Converts the frame to a tensor of shape [1, 3, 256, 128]
def image_to_tensor(image):
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
        v2.Resize((256, 128)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])

    # Apply transforms
    tensor = transforms(image)
    return tensor

def extract_features():
    """
    Crops a region from the specified frame and converts it into a normalized tensor.

    The output tensor has shape [3, 256, 128] and is normalized using ImageNet statistics.
    This format is typically used for appearance-based feature extraction models.

    Returns:
        None: Writes df to new csv file containing features.
    """
    print("Getting features for all detections...")

    with open(f"csv_output/has_features/features_{config.extractor_name}_{config.video_name}.csv", mode="w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'frame', 'x1', 'y1', 'x2', 'y2', 'features'])
        writer.writeheader()
        df = pd.read_csv(f"csv_output/no_features/{config.metadata_file_name}")
        batch_size = 1000

        # Iterate through dataframe in chunks of specified batch size
        for i in range(0, len(df), batch_size):

            # Create batch of 1000 rows from df
            batch_df = df.iloc[i:i + batch_size]
            
            filenames = batch_df['filename'].tolist()

            # Create list of PIL images from filenames
            images = [Image.open(f"crops/{config.video_name}/{filename}") for filename in filenames]

            # Iterate through images in mini-batch chunks of 64 for model inference
            for j in range(0, len(images), 64):

                # Create mini-batch if 64 images and convert each to tensor
                images_mini_batch = images[j:j + 64]
                tensor_list = [image_to_tensor(img) for img in images_mini_batch]

                # Inference if tensor_list is full
                if tensor_list:

                    # Batch tensor list and extract features
                    batch = torch.stack(tensor_list)
                    features = config.extractor(batch)

                    # Create mini-batch of 64 rows from already batched df
                    mini_batch_df = batch_df.iloc[j:j + 64]

                    # Iterate through every row in dataframe and every output from model(feature)
                    for (_, row), feature in zip(mini_batch_df.iterrows(), features):
                        writer.writerow({
                            'filename': row['filename'],
                            'frame': row['frame'],
                            'x1': row['x1'],
                            'y1': row['y1'],
                            'x2': row['x2'],
                            'y2': row['y2'],
                            'features': json.dumps(feature.tolist())
                        })
        print(f"Finished getting features for all detections, csv saved in csv_output/has_features/{config.video_name}.csv")
        f.close()