import cv2


def Yolo_predict(model, video_path, output_folder, SAVE_IMAGES=False):
    # Abrir el video
    cap = cv2.VideoCapture(video_path)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        exit()

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Si no hay más frames, salimos del bucle
        
        #Inferencia Yolo
        results = model(frame)

        # Obtener las boxes en formato xyxy
        boxes = results[0].boxes
        person_boxes = boxes[boxes.cls == 0]
        
        person_boxes = person_boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

        frame_filename = f"frame_{frame_count:04d}.txt"
        txt_path = f"{output_folder}/{frame_filename}"

        
        if SAVE_IMAGES:
            img_path = f"/home/darwonl/Escritorio/PROJECT/ACC/Modelos/YOLO/Yolov8/frames/frame_{frame_count:04d}.jpg"
            img_with_boxes = results[0].plot()
            cv2.imwrite(img_path, img_with_boxes)
        
        
        with open(txt_path, "w") as f:
            for box in person_boxes:
                x1, y1, x2, y2 = box[:4]
                f.write(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

        # Incrementar el contador de frames
        frame_count += 1

    # Liberar el objeto VideoCapture
    cap.release()

    print(f"Proceso terminado. Se extrajeron {frame_count} frames.")
