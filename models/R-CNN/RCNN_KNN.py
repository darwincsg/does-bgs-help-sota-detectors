import cv2
import torch
from pathlib import Path
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np


def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

def process_video(video_path: Path, output_dir: Path, predictor: DefaultPredictor):
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel = np.ones((2, 2), dtype=np.uint8)
    kernel2 = np.ones((5, 5), dtype=np.uint8)

    knn = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = knn.apply(frame)
        mask = cv2.erode(fg_mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel2, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)

        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        outputs = predictor(foreground)
        instances = outputs["instances"].to("cpu")
        
        # Clase 0 corresponde a 'person' en COCO
        person_instances = instances[instances.pred_classes == 0]

        boxes = person_instances.pred_boxes.tensor.numpy()

        txt_filename = output_dir / f"frame_{frame_id:06d}.txt"
        with open(txt_filename, "w") as f:
            for box in boxes:
                x1, y1, x2, y2 = box
                f.write(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

        print(f"[{video_path.name}] Frame {frame_id} - Persons: {len(boxes)}")
        frame_id += 1

    cap.release()

def main():
    input_dir = Path("./Raw/Raw")
    output_folder = Path("./RCNN/KNN")
    output_folder.mkdir(parents=True, exist_ok=True)

    predictor = load_model()

    for video_file in input_dir.glob("*.MP4"):
        print(f"\Processing video: {video_file.name}")
        video_output_dir = output_folder / video_file.stem
        process_video(video_file, video_output_dir, predictor)

    print("\n All video procesed.")

if __name__ == "__main__":
    main()
