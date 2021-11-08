import tensorflow as tf 
import numpy as np
import cv2
import trimesh
import pyrender

import os
import glob
import platform
import argparse
import time

from mtcnn import MTCNN
from PIL import Image
from scipy.io import loadmat,savemat
from preprocess_img import align_img
from utils import *
from face_decoder import Face3D
from options import Option

is_windows = platform.system() == "Windows"

#set up path
image_test_path = 'images'
input_path = 'input'
landmark_path = 'input'
reconstruct_path = 'output'
rasterize_path = 'render images'

if not os.path.exists(reconstruct_path):
	os.makedirs(reconstruct_path)

if not os.path.exists(rasterize_path):
	os.makedirs(rasterize_path)

#get image list
img_list =  glob.glob(input_path + '/' + '*.jpg')
img_list += glob.glob(input_path + '/' + '*.png')
# img_list += glob.glob(input_path + '/' + '*.JPG')
# img_list += glob.glob(input_path + '/' + '*.PNG')

def parse_args():

    desc = "Deep3DFaceReconstruction"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--pretrain_weights', type=str, default=None, help='path for pre-trained model')
    parser.add_argument('--use_pb', type=int, default=1, help='validation data folder')

    return parser.parse_args()

def restore_weights(sess,opt):
	var_list = tf.trainable_variables()
	g_list = tf.global_variables()

	# add batch normalization params into trainable variables 
	bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
	bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
	var_list +=bn_moving_vars

	# create saver to save and restore weights
	saver = tf.train.Saver(var_list = var_list)
	saver.restore(sess,opt.pretrain_weights)

def detect():
	detector = MTCNN()
	count = 0

	# img_list =  glob.glob(input_path + '/' + '*.jpg')
	# img_list += glob.glob(input_path + '/' + '*.png')
	# img_list += glob.glob(input_path + '/' + '*.JPG')
	# img_list += glob.glob(input_path + '/' + '*.PNG')

	for file in img_list:
		count += 1
		print(count)

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

		#save detected landmark as text file
		save_landmark(os.path.join(landmark_path,file.split(os.path.sep)[-1].replace('.png','.txt').replace('.PNG','.txt').replace('.jpg','.txt').replace('.JPG','.txt')), 
			left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, mouth_left_x, mouth_left_y, mouth_right_x, mouth_right_y) 

def reconstruct():
	count = 0
	#initialize tic
	tic = time.perf_counter()

	# input and output folder
	args = parse_args()
	
	# img_list =  glob.glob(input_path + '/' + '*.jpg')
	# img_list += glob.glob(input_path + '/' + '*.png')
	# img_list += glob.glob(input_path + '/' + '*.JPG')
	# img_list += glob.glob(input_path + '/' + '*.PNG')

	# read BFM face model
	# transfer original BFM model to our model
	if not os.path.isfile('./BFM/BFM_model_front.mat'):
		transferBFM09()

	# read standard landmarks for preprocessing images
	lm3D = load_lm3d()

	# build reconstruction model
	with tf.Graph().as_default() as graph:
		
		with tf.device('/cpu:0'):
			opt = Option(is_train=False)
		opt.batch_size = 1
		opt.pretrain_weights = args.pretrain_weights
		FaceReconstructor = Face3D()
		images = tf.placeholder(name = 'input_imgs', shape = [opt.batch_size,224,224,3], dtype = tf.float32)

		if args.use_pb and os.path.isfile('network/FaceReconModel.pb'):
			print('Using pre-trained .pb file.')
			graph_def = load_graph('network/FaceReconModel.pb')
			tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})
			# output coefficients of R-Net (dim = 257) 
			coeff = graph.get_tensor_by_name('resnet/coeff:0')
		else:
			print('Using pre-trained .ckpt file: %s'%opt.pretrain_weights)
			import networks
			coeff = networks.R_Net(images, is_training=False)

		# reconstructing faces
		FaceReconstructor.Reconstruction_Block(coeff,opt)
		
		face_shape = FaceReconstructor.face_shape #frontal view
		# face_shape = FaceReconstructor.face_shape_t #original pose
		face_texture = FaceReconstructor.face_texture
		face_color = FaceReconstructor.face_color
		landmarks_2d = FaceReconstructor.landmark_p
		recon_img = FaceReconstructor.render_imgs
		tri = FaceReconstructor.facemodel.face_buf

		# print("Face Texture: ", face_texture, "\n")
		# print("Face Color: ", face_color, "\n")
		# print("Landmarks 2D: ", landmarks_2d, "\n")

		with tf.Session() as sess:

			#print("Face Shape: ", sess.run(face_shape), "\n")

			if not args.use_pb :
				restore_weights(sess,opt)

			print('reconstructing...')
			for file in img_list:
				count += 1
				print(count)

				# load images and corresponding 5 facial landmarks
				img,lm = load_img(file,file.replace('png','txt').replace('PNG','txt').replace('jpg','txt').replace('JPG','txt'))
				# preprocess input image
				input_img, lm_new, transform_params = align_img(img,lm,lm3D)

				#save input images after cropped 
				# input_img = np.squeeze(input_img, (0))
				# cv2.imwrite(os.path.join(image_test_path, file.split(os.path.sep)[-1]), input_img)
				# input_img = np.expand_dims(input_img,0)

				coeff_,face_shape_,face_texture_,face_color_,landmarks_2d_,recon_img_,tri_ = sess.run([coeff,\
					face_shape,face_texture,face_color,landmarks_2d,recon_img,tri],feed_dict = {images: input_img})

				# print("Coefficients: ", coeff_, "\n")

				# reshape outputs

				# print("Face Shape Before Squeeze: ", face_shape_, "\n")
				# print("Face Texture Before Squeeze: ", face_texture_, "\n")
				# print("Face Color Before Squeeze: ", face_color_, "\n")
				# print("Landmarks 2D Before Squeeze: ", landmarks_2d_, "\n")

				input_img = np.squeeze(input_img)
				face_shape_ = np.squeeze(face_shape_, (0))
				face_texture_ = np.squeeze(face_texture_, (0))
				face_color_ = np.squeeze(face_color_, (0))
				landmarks_2d_ = np.squeeze(landmarks_2d_, (0))

				# print("Face Shape After Squeeze: ", face_shape_, "\n")
				# print("Face Texture After Squeeze: ", face_texture_, "\n")
				# print("Face Color After Squeeze: ", face_color_, "\n")
				# print("Landmarks 2D After Squeeze: ", landmarks_2d_, "\n")

				# save output files
				if not is_windows:
					recon_img_ = np.squeeze(recon_img_, (0))
					cv2.imwrite(os.path.join(image_test_path, file.split(os.path.sep)[-1]), recon_img_)
					savemat(os.path.join(reconstruct_path,file.split(os.path.sep)[-1].replace('.png','.mat').replace('.PNG','.mat').replace('jpg','mat').replace('JPG','mat')),{'cropped_img':input_img[:,:,::-1],'recon_img':recon_img_,'coeff':coeff_,\
						'face_shape':face_shape_,'face_texture':face_texture_,'face_color':face_color_,'lm_68p':landmarks_2d_,'lm_5p':lm_new})
				
				save_obj(os.path.join(reconstruct_path,file.split(os.path.sep)[-1].replace('.png','_mesh.obj').replace('.PNG','_mesh.obj').replace('.jpg','_mesh.obj').replace('.JPG','_mesh.obj')),face_shape_,tri_,np.clip(face_color_,0,255)/255) # 3D reconstruction face (in canonical view)
				save_shape_txt(os.path.join(reconstruct_path,file.split(os.path.sep)[-1].replace('.png','_shape.txt').replace('.PNG','_shape_txt').replace('.jpg','_shape.txt').replace('.JPG','_shape.txt')),face_shape_,tri_,np.clip(face_color_,0,255)/255)
	
	#return timer
	toc = time.perf_counter()
	print(f"Created meshes in {toc - tic:0.4f} seconds")

def rasterize():
	object_list = glob.glob(reconstruct_path + '/' + '*.obj')

	#set up pyrender scene
	scene = pyrender.Scene(bg_color = [0.0, 0.0, 0.0])

	#set up pyrender camera
	def camera_pose():
		centroid = [0.0, 0.0, 2.5]
		cp = np.eye(4)
		s2 = 1.0 / np.sqrt(2.0)

		cp[:3,:3] = np.array([
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0]
		])

		hfov = np.pi / 3.0
		dist = scene.scale / (2.0 * np.tan(hfov))
		cp[:3,3] = dist * np.array([0.0, 0.0, 0.0]) + centroid

		return cp

	pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.5, znear=0.5, zfar=50.0)
	oc = pyrender.OrthographicCamera(xmag = 1.0, ymag = 1.0, znear = 0.5, zfar = 50.0)
	npc = pyrender.Node(matrix=camera_pose(), camera=pc)
	noc = pyrender.Node(matrix=camera_pose(), camera=oc)

	scene.add_node(npc)
	scene.add_node(noc)
	#noc = orthographic, npc = perspective
	scene.main_camera_node = npc

	#set up pyrender light
	dlight = pyrender.DirectionalLight(color=[0.9, 0.75, 0.7], intensity=5.0)

	scene.add(dlight, pose=camera_pose())

	#load mesh
	count = 0

	for file in object_list:
		count += 1
		print(count)
		tic1 = time.perf_counter()
		input_mesh = trimesh.load(file)
		mesh = pyrender.Mesh.from_trimesh(input_mesh)
		nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
		scene.add_node(nm)
		
		r = pyrender.OffscreenRenderer(viewport_width=1024, viewport_height=1024, point_size=1.0)
		
		color, depth = r.render(scene)
		
		b,g,red = cv2.split(color)
		result = cv2.merge((red,g,b))
		cv2.imwrite(os.path.join(rasterize_path, file.split(os.path.sep)[-1].replace('.obj','.png')), result)
		
		scene.remove_node(nm)
		
		r.delete()

if __name__ == '__main__':
	detect()
	reconstruct()
	rasterize()
