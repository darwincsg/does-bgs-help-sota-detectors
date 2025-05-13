#import argparse
from ultralytics import YOLO
from pathlib import Path
import os

from scripts.file_functions import convert_cvat_xml_to_abs
from scripts.models import Yolo_predict
from scripts.IoU_function import get_Average



def build(video_path):
    video_split = video_path.strip().split('/')

    video_file = Path(video_split[len(video_split)-1])
    absolute_name = video_file.stem
    
    model_name = "yolov8x.pt"
    model = YOLO(model_name)

    #CREATE XML DIRECTORY AND FILE NAME
    Xml_Path = f'/home/darwonl/Escritorio/PROJECT/ACC/Modelos/YOLO/Yolov8/xml/{absolute_name}'
    os.makedirs(Xml_Path, exist_ok=True)
    Xml_file = f'/home/darwonl/Escritorio/PROJECT/ACC/Modelos/Detection/Outdoor/Outdoor/{absolute_name}.xml'
    
    #CREATE YOLO DIRCTORY
    Yolo_Path = f'/home/darwonl/Escritorio/PROJECT/ACC/Modelos/YOLO/Yolov8/yolo/{absolute_name}'
    os.makedirs(Yolo_Path, exist_ok=True)
    
    #First part, we sent the xml and convert it to txt files on the output dir
    convert_cvat_xml_to_abs(Xml_file, Xml_Path)
    
    #Second part, call Yolo and made the predictions
    Yolo_predict(model,video_path, Yolo_Path)
    
    #Final part, do the comparison!
    final = get_Average(Xml_Path,Yolo_Path)
    
    return final


def main():

    path_video = '/home/darwonl/Escritorio/PROJECT/ACC/Modelos/Detection/Raw/Raw/099_012_07_05_2024_10_15_SRM_5.8_10_30.MP4'

    final = build(path_video)

    print(final)





if __name__ == "__main__":
    main()
