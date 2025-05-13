import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.file_functions import function_on_file

def final_accuracies(matches, matrix,size):
    accuracy = 0
    for i, j in matches:
        accuracy += matrix[i][j]
    if(size!=0):
        return accuracy/size
    return 0


#Open the file and create the lists
def create_coordsList(listN):
    Coords = []
    
    for coords in listN:
        line = coords.strip().split()
        line = [float(element) for element in line] 
        #x_center, y_center, width, height = line[0:4]
        Coords.append(line)
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
    """
    boxes_A y boxes_B son listas de bounding boxes en formato [x_min, y_min, x_max, y_max]
    Retorna una matriz (len(A) x len(B)) donde cada celda es el IoU entre una box de A y una de B
    """
    iou_matrix = np.zeros((len(boxes_A), len(boxes_B)))

    for i, box_A in enumerate(boxes_A):
        for j, box_B in enumerate(boxes_B):
            iou_matrix[i, j] = IoU(box_A, box_B)

    return iou_matrix

def match_boxes_optimal(iou_matrix, iou_threshold=0.5):
    if iou_matrix.size == 0:
        return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))

    # Convertimos IoU a costo: mayor IoU = menor costo (para el algoritmo de minimización)
    cost_matrix = 1 - iou_matrix

    # Aplicamos asignación óptima
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_indices = []
    unmatched_A = list(range(iou_matrix.shape[0]))
    unmatched_B = list(range(iou_matrix.shape[1]))

    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            matched_indices.append((i, j))
            unmatched_A.remove(i)
            unmatched_B.remove(j)

    return matched_indices, unmatched_A, unmatched_B


def compare_annotations(pathX, pathY):
    
    List_Grand = function_on_file(pathX)
    List_Yolo = function_on_file(pathY)

    Coords_G = create_coordsList(List_Grand)
    Coords_Y = create_coordsList(List_Yolo)
    
    #Create the matrix, first we put Grand True list and then Yolo list
    matrix = compute_iou_matrix(Coords_G,Coords_Y)

    results = match_boxes_optimal(matrix)

    original_size = len(List_Grand)
    
    TP = len(results[0])
    FP = len(results[1])
    FN = len(results[2])
        
    return TP, FP, FN

def get_Average(path_grand_true, path_annotations):
    gt_files = sorted(os.listdir(path_grand_true))
    yolo_files = sorted(os.listdir(path_annotations))
    
    # Inicializa acumuladores
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for gt_file, yolo_file in zip(gt_files, yolo_files):
        # Crea los paths absolutos
        path_G = os.path.join(path_grand_true, gt_file)
        path_Y = os.path.join(path_annotations, yolo_file)
            
        accuracy = compare_annotations(path_G,path_Y)

        # Acumular
        total_tp += accuracy[0]
        total_fp += accuracy[1]
        total_fn += accuracy[2]

    #Precisio - recall - f1_score calculate
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'Precision': precision,
        'Recall': recall,
        'F1': f1_score
    }
