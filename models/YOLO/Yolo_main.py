import cv2
import os
from pathlib import Path
from ultralytics import YOLO  

#YOLO main function, work in Yolov8 and return txt's with arbitrary coordinates of bounding boxes.
def Yolo_predict(model, video_path, output_folder, SAVE_IMAGES=False):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Can't open the video {video_path}")
        return

    frame_count = 0
    video_stem = Path(video_path).stem 

    output_video_dir = output_folder / video_stem
    output_video_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame,conf=0.2)
        boxes = results[0].boxes
        person_boxes = boxes[boxes.cls == 0]
        person_boxes = person_boxes.xyxy.cpu().numpy()

        txt_path = output_video_dir / f"frame_{frame_count:06d}.txt"
        with open(txt_path, "w") as f:
            for box in person_boxes:
                x1, y1, x2, y2 = box[:4]
                f.write(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

        if SAVE_IMAGES:
            img_path = output_video_dir / f"frame_{frame_count:06d}.jpg"
            img_with_boxes = results[0].plot()
            cv2.imwrite(str(img_path), img_with_boxes)

        print(f"[{video_stem}] Frame {frame_count} - Persons: {len(person_boxes)}")
        frame_count += 1

    cap.release()

def main():

    input_dir = Path("./Raw/Raw")
    output_folder = Path("./YOLO/Yolo")
    output_folder.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8x.pt")  # Yolov8 model, can be changed

    for video_path in sorted(input_dir.glob("*.MP4")):
        
        print(f"Processing video: {video_path.name}")
        Yolo_predict(model, video_path, output_folder, SAVE_IMAGES=False)


if __name__ == "__main__":
    main()
