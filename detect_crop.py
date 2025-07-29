import ultralytics
from ultralytics import YOLO
from PIL import Image
import csv
import consts
import cv2 as cv

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
    print("Making detections and cropping bounding boxes...")
    # Load yolo model
    model = YOLO(r"C:\Users\VrTeleop-01\Documents\bjj_project\jiu_jitsu_detection_v2.pt")
    cap = cv.VideoCapture(consts.video_path)

    last_frame_num = cap.get(cv.CAP_PROP_FRAME_COUNT)
    nth_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if nth_frame % 10 == 0 or nth_frame == last_frame_num:

            # Detect people and crop detections
            results = model.predict(
                source=frame,
                stream=False,
                classes=[0]
            )
            process_detections(results, frame_idx = nth_frame)
        nth_frame += 1

    cap.release()
    cv.destroyAllWindows()
    print(f"Finished getting bbox data for all detections, csv saved in csv_output/no_features/{consts.metadata_file_name}")

def process_detections(results, frame_idx):
    # Write detections to csv file
    with open(f"csv_output/no_features/{consts.metadata_file_name}", mode="a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'frame', 'x1', 'y1', 'x2', 'y2'])
        writer.writeheader()

        # Iterate through every detection in frame
        for result in results:

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
