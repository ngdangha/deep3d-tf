import cv2
import os 
import glob

input_path = 'blur-dataset'
output_path = 'input'

for folder in os.scandir(input_path):
    img_list = glob.glob(folder.path + '/' + '*.png')
    img_list +=glob.glob(folder.path + '/' + '*.jpg')
    img_list +=glob.glob(folder.path + '/' + '*.PNG')
    img_list +=glob.glob(folder.path + '/' + '*.JPG')

    n = 0
    name = folder.name + '_'

    for file in img_list:
        n += 1
        print(n)
        img = cv2.imread(file)
        cv2.imwrite(os.path.join(output_path, name + file.split(os.path.sep)[-1]), img)
        # os.remove(file)