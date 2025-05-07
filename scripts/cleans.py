import os

def clean_txt_of_xml(path):
    with open(path, "r+") as file:
        lines = file.readlines()      
        
        new_content = []
        
        for x in lines:
            split_string = x.strip().split()
            new_string = f'0 {split_string[1]} {split_string[2]} {split_string[3]} {split_string[4]}\n'
            new_content.append(new_string)
        
        file.seek(0)
        file.writelines(new_content)
        file.truncate()  
        
def clean_txt_of_yolo(path):
    with open(path, "r+") as file:
        lines = file.readlines()

        filtered_lines = []
        for x in lines:
            split_string = x.strip().split()
            if split_string[0] == '0':
                filtered_lines.append(x)

        # Override file
        file.seek(0)
        file.writelines(filtered_lines)
        file.truncate()
        
        
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
