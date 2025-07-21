import ultralytics
from ultralytics import YOLO
from PIL import Image
import csv
import consts

def crop_img(frame, x1, y1, x2, y2):
    """
    Crops a region from the specified frame.

    Parameters:
        frame (int): The frame number to retrieve and crop from the video.
        x1 (int): Top-left x-coordinate of the bounding box in pixels.
        y1 (int): Top-left y-coordinate of the bounding box in pixels.
        x2 (int): Bottom-right x-coordinate of the bounding box in pixels.
        y2 (int): Bottom-right y-coordinate of the bounding box in pixels.

    Returns:
        np array: Shape [H, W, 3].
    """
    crop = frame[y1:y2, x1:x2]
    crop_rgb = crop[..., ::-1]
    return crop_rgb

# Make detections with yolo model
def make_detections():

    # Load yolo model
    model = YOLO('yolov8n.pt')

    # Detect people
    results = model.predict(
        source=consts.video_path,
        stream=True,
        classes=[0]
    )
    
    process_detections(results)

def process_detections(results):
    print("Detecting people and cropping bounding boxes...")
    # Write detections to csv file
    with open(f"csv_output/no_features/{consts.metadata_file_name}", mode="w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'frame', 'x1', 'y1', 'x2', 'y2'])
        writer.writeheader()

        # Iterate through every result in frame
        for frame_idx, result in enumerate(results):

            #Store current frame for later cropping
            frame = result.orig_img

            # Iterate through every box(detection) in frame and assign id
            for i, box in enumerate(result.boxes):

                # Get box dimensions for cropping and crop
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                crop = crop_img(frame, x1, y1, x2, y2)

                # Give image of crop unique file name where frame_index=current frame and i=current box in frame
                filename = f"f{frame_idx}_d{i}.jpg"
                Image.fromarray(crop).save(f"crops/{consts.video_name}/{filename}")

                # Write extracted box features to csv file
                writer.writerow({
                    'filename': filename,
                    'frame': frame_idx,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                })
                print(f"Finished getting features for all detections, csv saved in csv_output/no_features/{consts.metadata_file_name}")
