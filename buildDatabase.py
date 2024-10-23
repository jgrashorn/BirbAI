import csv
import os
import cv2

def build_csv(directory_string, output_csv_name):
    """Builds a csv file for pytorch training from a directory of folders of images.
    Install csv module if not already installed.
    Args: 
    directory_string: string of directory path, e.g. r'.\data\train'
    output_csv_name: string of output csv file name, e.g. 'train.csv'
    Returns:
    csv file with file names, file paths, class names and class indices
    """
    directory = directory_string
    class_lst = os.listdir(directory) #returns a LIST containing the names of the entries (folder names in this case) in the directory.
    class_lst.sort() #IMPORTANT 
    print(class_lst)

    with open(output_csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['file_name', 'file_path', 'class_name', 'class_index']) #create column names
        for class_name in class_lst:
            print(class_name)
            class_path = os.path.join(directory, class_name) #concatenates various path components with exactly one directory separator (‘/’) except the last path component. 
            file_list = os.listdir(class_path) #get list of files in class folder
            for file_name in file_list:
                file_path = os.path.join(directory, class_name, file_name) #concatenate class folder dir, class name and file name
                
                with open(file_path, 'rb') as im :
                    im.seek(-2,2)
                    if not im.read() == b'\xff\xd9':
                        # fix image
                        img = cv2.imread(file_path)
                        cv2.imwrite(file_path, img)
                        print('FIXED corrupted image :', file_path) 

                try:
                    writer.writerow([file_name, file_path, class_name, class_lst.index(class_name)]) #write the file path and class name to the csv file
                except:
                    print(file_name)
                    print(file_path)
    return

if __name__=="__main__":
    build_csv("birb/","data.csv")