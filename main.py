import os
import csv
from scripts.IoU_function import get_Average

# Directories Path
ground_truth_root = "./Ground_True"
annotations_root = "./Annotations"
output_file = "./Example_output/Results.csv"

common_folders = sorted(set(os.listdir(ground_truth_root)).intersection(set(os.listdir(annotations_root))))
resultados = []

for folder_name in common_folders:
    gt_folder = os.path.join(ground_truth_root, folder_name)
    ann_folder = os.path.join(annotations_root, folder_name,'txt')

    if os.path.isdir(gt_folder) and os.path.isdir(ann_folder):
        resultados_metricas = get_Average(gt_folder, ann_folder)
        resultados.append({
            'Folder': folder_name,
            **resultados_metricas
        })
        print(f'Video {folder_name}: Processed')

# Check if the file already exists
archivo_existe = os.path.exists(output_file)

# Save in append mode and write header only if file does not exist
with open(output_file, mode='a', newline='') as csvfile:
    fieldnames = ['Folder', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    if not archivo_existe:
        writer.writeheader()

    for fila in resultados:
        writer.writerow(fila)

print(f"Results added to: {output_file}")