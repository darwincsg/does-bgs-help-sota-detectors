import xml.etree.ElementTree as ET
import os
import cv2

def function_on_file(path):
    with open(path, "r") as file:
        lines = fike.readlines()        
        lines.sort()
        return lines

def get_image_size(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open frame: {image_path}")
    height, width = img.shape[:2]
    return width, height

def convert_cvat_xml_to_yolo(xml_path, output_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    meta = root.find('meta')
    width = int(meta.find('original_size/width').text)
    height = int(meta.find('original_size/height').text)

    label_names = meta.find('job/labels')
    label_list = [label.find('name').text for label in label_names.findall('label')]
    label_list = sorted(set(label_list))  # unique & sorted for consistency
    label_to_class = {label_name: i for i, label_name in enumerate(label_list)}

    frame_dict = {}

    for track in root.findall('track'):
        label_name = track.get('label')  # e.g. '0497'
        class_id = label_to_class[label_name]

        # For each box in this track
        for box in track.findall('box'):
            outside = int(box.get('outside'))
            # If outside==1, the object is no longer visible, so skip
            if outside == 1:
                continue

            frame_idx = int(box.get('frame'))
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # Convert to YOLO
            bbox_width = xbr - xtl
            bbox_height = ybr - ytl
            x_center = xtl + bbox_width / 2.0
            y_center = ytl + bbox_height / 2.0

            # Normalize [0,1]
            x_center_norm = x_center / width
            y_center_norm = y_center / height
            w_norm = bbox_width / width
            h_norm = bbox_height / height

            yolo_line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

            # Accumulate the line in frame_dict
            if frame_idx not in frame_dict:
                frame_dict[frame_idx] = []
            frame_dict[frame_idx].append(yolo_line)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for frame_idx, lines in frame_dict.items():
        out_txt_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.txt")
        with open(out_txt_path, 'w') as f:
            f.write("\n".join(lines))

    print(f"Conversion complete! YOLO annotations saved to: {output_dir}")
