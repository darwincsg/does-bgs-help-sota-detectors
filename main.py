from scripts.cleans import clean_directory, clean_txt_of_xml, clean_txt_of_yolo
from scripts.file_functions import convert_cvat_xml_to_yolo
from scripts.IoU_function import get_Average
from pathlib import Path
import argparse

def main(arg_XML, arg_YOLO, arg_Frame):
    path_XML = arg_XML.strip().split('/')

    archivo_Xml = Path(path_XML[len(path_XML)-1])
  
    if archivo_Xml.suffix == ".xml":
        absolute_name = archivo_Xml.stem
        directory_name = f'/home/darwonl/Escritorio/PROJECT/ACC/Modelos/YOLO/Labels/{absolute_name}'
    else:
        raise ValueError("El archivo del primer argumento no es un XML válido.")

    convert_cvat_xml_to_yolo(arg_XML,directory_name)

    clean_directory(directory_name,'XML')

    clean_directory(arg_YOLO,'YOLO')

    final = get_Average(directory_name,arg_YOLO,arg_Frame)

    print(final)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to obtain the total accuracy of a Yolo annotation")

    parser.add_argument("arg1", type=str, help="Grand True path")
    parser.add_argument("arg2", type=str, help="Yolo annotatio path")
    parser.add_argument("arg3", type=str, help="Frame path")

    args = parser.parse_args()
    
    main(args.arg1, args.arg2, args.arg3)