import cv2
import numpy as np
from pathlib import Path

def filter_boxes_with_mask(boxes, mask, threshold=0.1):
    filtered = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(mask.shape[1] - 1, x2), min(mask.shape[0] - 1, y2)
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        non_zero = cv2.countNonZero(roi)
        if roi.shape[0] * roi.shape[1] > 0 and non_zero / (roi.shape[0] * roi.shape[1]) >= threshold:
            filtered.append((x1, y1, x2, y2))
    return filtered

def process_video(video_path, txt_dir, output_dir, threshold=0.1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Can't open the video: {video_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    kernel = np.ones((2, 2), dtype=np.uint8)
    kernel2 = np.ones((4, 4), dtype=np.uint8)

    mog2 = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True)
    mog2.setShadowThreshold(0.5)

    frame_id = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        txt_file = txt_dir / f"frame_{frame_id:06d}.txt"
        if not txt_file.exists():
            frame_id += 1
            continue

        with open(txt_file, "r") as f:
            boxes = [list(map(float, line.strip().split())) for line in f]

        fg_mask = mog2.apply(frame)
        mask = cv2.erode(fg_mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel2, iterations=2)
        mask = cv2.erode(mask, kernel2, iterations=1)

        filtered_boxes = filter_boxes_with_mask(boxes, mask, threshold)

        output_file = output_dir / f"frame_{frame_id:06d}.txt"
        with open(output_file, "w") as f:
            for x1, y1, x2, y2 in filtered_boxes:
                f.write(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

        print(f"[{video_path.name}] Frame {frame_id} - Filtered: {len(filtered_boxes)}")
        frame_id += 1

    cap.release()
    print(f"Video finished: {video_path.name}")

def main():
    videos_dir = Path("./Raw/Raw")
    txts_dir = Path("./RCNN/Rcnn") #Any folder created by the models can be placed
    output_base_dir = Path("./RCNN/MOG2_FIILTERED")

    for video_path in videos_dir.glob("*.MP4"):
        video_name = video_path.stem
        txt_dir = txts_dir / video_name 
        output_dir = output_base_dir / video_name / "txt"

        if not txt_dir.exists():
            print(f"Folder TXT does not exist: {video_name}")
            continue

        print(f"\Processing video: {video_name}")
        process_video(video_path, txt_dir, output_dir)

    print("\nAll videos processed.")

if __name__ == "__main__":
    main()
