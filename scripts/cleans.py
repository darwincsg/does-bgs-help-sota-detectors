import os

def clean_txt_of_xml(path):
    with open(path, "r+") as archivo:
        linhas = archivo.readlines()      
        
        new_content = []
        
        for x in linhas:
            split_string = x.strip().split()
            new_string = f'0 {split_string[1]} {split_string[2]} {split_string[3]} {split_string[4]}\n'
            new_content.append(new_string)
        
        archivo.seek(0)
        archivo.writelines(new_content)
        archivo.truncate()  
        
def clean_txt_of_yolo(path):
    with open(path, "r+") as archivo:
        linhas = archivo.readlines()

        linhas_filtradas = []
        for x in linhas:
            split_string = x.strip().split()
            if split_string[0] == '0':
                linhas_filtradas.append(x)

        # Sobrescribimos el archivo
        archivo.seek(0)
        archivo.writelines(linhas_filtradas)
        archivo.truncate()
        
        
def clean_directory(path, option):
    files_list = sorted(os.listdir(path))
    
    for actual_file in files_list:
        path_Abs = os.path.join(path, actual_file)
        if(option == 'XML'):
            clean_txt_of_xml(path_Abs)    
        elif (option == 'YOLO'):
            clean_txt_of_yolo(path_Abs)
        else:
            break
