import numpy as np
import os
import glob

def save_obj(path,v,f):
	with open(path,'w') as file:
		for i in range(len(v)):
			file.write('v %f %f %f\n'%(v[i,0],v[i,1],v[i,2]))

		file.write('\n')

		for i in range(len(f)):
			file.write('f %d %d %d\n'%(f[i,0],f[i,1],f[i,2]))

	file.close()

def save_shape_np(path, s):
	with open(path,'w') as file:
		for i in range(len(s)):
			file.write('%f %f %f\n'%(s[i,0],s[i,1],s[i,2]))

		file.write('\n')

	file.close()

def save_shape_tf(path, s):
	with open(path,'w') as file:
		for i in range(107127):
			file.write('%f '%(s[0, i]))

	file.close()

#set up input and output path
input_path = 'output'
mean_path = 'mean'

if not os.path.exists(mean_path):
    os.makedirs(mean_path)

#set up variables
count = 0
sum_matrix = np.zeros((35709, 3))
indices_matrix = np.loadtxt('mean/pos.txt', dtype='f', delimiter= ' ')

#main process
shape_list = glob.glob(input_path + '/' + '*.txt')

for file in shape_list:
    count += 1
    new_matrix = np.loadtxt(file, dtype='f', delimiter= ' ')
    sum_matrix = sum_matrix + new_matrix
    print(count)

#calculate mean shape
mean_matrix = sum_matrix/count
# tf_matrix = np.reshape(mean_matrix, (-1, 107127))

# print(np.shape(tf_matrix))

#save mean shape as object file
save_shape_np(mean_path + '/' + str(count) + '_mean.txt', mean_matrix)
# save_shape_tf(mean_path + '/' + str(count) + '_mean.txt', tf_matrix)
save_obj(mean_path + '/' + str(count) + '_mean.obj', mean_matrix, indices_matrix)
