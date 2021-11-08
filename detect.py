import cv2

import os 
import glob
import time

from mtcnn import MTCNN

#initialize tic
tic = time.perf_counter()

#set up input, output directory
image_path = 'input'
landmark_path = 'input'

if not os.path.exists(landmark_path):
    os.makedirs(landmark_path)

img_list = glob.glob(image_path + '/' + '*.png')
img_list +=glob.glob(image_path + '/' + '*.PNG')
img_list +=glob.glob(image_path + '/' + '*.jpg')
img_list +=glob.glob(image_path + '/' + '*.JPG')

#set up detector
detector = MTCNN()

#save landmarks as textfile
def save_landmark(path, lex, ley, rex, rey, nx, ny, mlx, mly, mrx, mry):
	with open(path,'w') as file:
		file.write('%s\t%s\n'%(lex, ley))
		file.write('%s\t%s\n'%(rex, rey))
		file.write('%s\t%s\n'%(nx, ny))
		file.write('%s\t%s\n'%(mlx, mly))
		file.write('%s\t%s\n'%(mrx, mry))

	file.close()

#main process
n = 0

for file in img_list:
	n += 1
	print(n)
	img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    # face = detector.detect_faces(img)
	keypoints = detector.detect_faces(img)[0]['keypoints']

	left_eye_x = keypoints["left_eye"][0]
	left_eye_y = keypoints["left_eye"][1]

	right_eye_x = keypoints["right_eye"][0]
	right_eye_y = keypoints["right_eye"][1]

	nose_x = keypoints["nose"][0]
	nose_y = keypoints["nose"][1]

	mouth_left_x = keypoints["mouth_left"][0]
	mouth_left_y = keypoints["mouth_left"][1]

	mouth_right_x = keypoints["mouth_right"][0]
	mouth_right_y = keypoints["mouth_right"][1]

	save_landmark(os.path.join(landmark_path,file.split(os.path.sep)[-1].replace('.png','.txt').replace('.PNG','.txt').replace('.jpg','.txt').replace('.JPG','.txt')), 
		left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, mouth_left_x, mouth_left_y, mouth_right_x, mouth_right_y) 

#return timer
toc = time.perf_counter()
print(f"Detected landmarks in {toc - tic:0.4f} seconds")