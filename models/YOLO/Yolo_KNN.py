import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse

def yolo_predict(model, frame, frame_id, output_folder, save_images=False, output_image_dir=None, video_name="video"):
    results = model(frame, conf=0.2)
    boxes = results[0].boxes
    person_boxes = boxes[boxes.cls == 0]
    person_boxes = person_boxes.xyxy.cpu().numpy()

    output_folder.mkdir(parents=True, exist_ok=True)
    txt_path = output_folder / f"frame_{frame_id:06d}.txt"
    with open(txt_path, "w") as f:
        for box in person_boxes:
            x1, y1, x2, y2 = box[:4]
            f.write(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

    if save_images and output_image_dir:
        output_image_dir.mkdir(parents=True, exist_ok=True)
        img_path = output_image_dir / f"frame_{frame_id:06d}.jpg"
        img_with_boxes = results[0].plot()
        cv2.imwrite(str(img_path), img_with_boxes)

    print(f"[{video_name}] Frame {frame_id} - Personas detectadas: {len(person_boxes)}")

def process_video(video_path, model, output_base_path, save_images=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error al abrir video: {video_path}")
        return

    video_name = video_path.stem
    output_txt_dir = output_base_path / video_name / "txt"
    output_img_dir = output_base_path / video_name / "frames" if save_images else None

    output_txt_dir.mkdir(parents=True, exist_ok=True)

    kernel = np.ones((2, 2), dtype=np.uint8)
    kernel2 = np.ones((5, 5), dtype=np.uint8)

    knn = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    frame_id = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        fg_mask = knn.apply(frame)
        mask = cv2.erode(fg_mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel2, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)

        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        yolo_predict(model, foreground, frame_id, output_txt_dir, save_images, output_img_dir, video_name)
        frame_id += 1

    cap.release()
    print(f"Finalizado: {video_name} - Total de frames: {frame_id}")

def main():

    input_dir = Path('./Raw/Raw')
    output_folder = Path('./YOLO/MOG_2/')
    output_folder.mkdir(parents=True, exist_ok=True)

    model_path = "yolov8x.pt"

    model = YOLO(model_path)

    for video_path in input_dir.glob("*.MP4"):
        print(f"Processing video: {video_path.name}")
        process_video(video_path, model, output_folder, save_images=False)

if __name__ == "__main__":
    main()
