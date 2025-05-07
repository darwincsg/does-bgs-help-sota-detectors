from scripts.file_functions import function_on_file, get_image_size
import os
import numpy as np

def final_accuracies(matches, matrix,size):
    accuracy = 0
    for i, j in matches:
        accuracy += matrix[i][j]
    return accuracy/size

def yolo_2_coords(yolo_line, img_width=3840, img_height=2160):
    parts = list(map(float, yolo_line.strip().split()))
    x_center, y_center, width, height = parts[1:5]
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [x1, y1, x2, y2]

#Open the file and create the lists
def create_coordsList(listN):
    Coords = []
    
    for coords in listN:
        aux_list = yolo_2_coords(coords)
        Coords.append(aux_list)
    return Coords
    
def IoU(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Coordenadas de la intersección
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Cálculo del área de intersección
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Cálculo del área de cada bounding box
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Cálculo del área de la unión
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0  # Evita división por cero

    iou = inter_area / union_area
    return iou


def compute_iou_matrix(boxes_A, boxes_B):
    iou_matrix = np.zeros((len(boxes_A), len(boxes_B)))

    for i, box_A in enumerate(boxes_A):
        for j, box_B in enumerate(boxes_B):
            iou_matrix[i, j] = IoU(box_A, box_B)

    return iou_matrix

def match_boxes(iou_matrix, iou_threshold=0.5):

    matched_index = []
    used_A = set()
    used_B = set()

    # Greedy matching
    while True:
        max_iou = -1
        max_pair = (-1, -1)
        for i in range(iou_matrix.shape[0]):
            if i in used_A:
                continue
            for j in range(iou_matrix.shape[1]):
                if j in used_B:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_pair = (i, j)

        if max_iou < iou_threshold:
            break

        i, j = max_pair
        matched_index.append((i, j))
        used_A.add(i)
        used_B.add(j)

    unmatched_A = [i for i in range(iou_matrix.shape[0]) if i not in used_A]
    unmatched_B = [j for j in range(iou_matrix.shape[1]) if j not in used_B]

    return matched_index, unmatched_A, unmatched_B

def compare_annotations(pathX, pathY, pathFrame):
    image_size = get_image_size(pathFrame)
    
    List_Grand = function_on_file(pathX)
    List_Yolo = function_on_file(pathY)

    Coords_G = create_coordsList(List_Grand)
    Coords_Y = create_coordsList(List_Yolo)

    #Create the matrix, first we put Grand True list and then Yolo list
    matrix = compute_iou_matrix(Coords_G,Coords_Y)

    final_list = match_boxes(matrix)

    matches = final_list[0]
    original_size = len(List_Grand)
    
    accuracy = final_accuracies(matches,matrix,original_size)
    
    return accuracy

def get_Average(path_grand_true, path_annotations, path_frames):
    gt_files = sorted(os.listdir(path_grand_true))
    yolo_files = sorted(os.listdir(path_annotations))

    # Init variables
    total_accuracy = 0
    total_files = 0

    for gt_file, yolo_file in zip(gt_files, yolo_files):
        # Create the absolute paths
        path_G = os.path.join(path_grand_true, gt_file)
        path_Y = os.path.join(path_annotations, yolo_file)

        accuracy = compare_annotations(path_G,path_Y,path_frames)

        # Increase variables
        total_accuracy += accuracy
        total_files += 1

        #print(f"{gt_file} -> Accuracy: {accuracy:.4f}")
    return total_accuracy / total_files
        
