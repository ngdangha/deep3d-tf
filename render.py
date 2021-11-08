import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import glob
import time

import trimesh
import pyrender
import numpy as np
import cv2

#initialize tic
# tic = time.perf_counter()

#set up input, output directory
object_path = 'output'
image_path = 'render images'

if not os.path.exists(image_path):
    os.makedirs(image_path)

object_list = glob.glob(object_path + '/' + '*.obj')

#set up pyrender scene
scene = pyrender.Scene(bg_color = [0.0, 0.0, 0.0])
# scene = pyrender.Scene()

#set up pyrender camera
def camera_pose():
    # centroid = scene.centroid
    centroid = [0.0, 0.0, 2.5]
    # centroid = [0.0, 0.0, 0.0]
    # print(centroid)
    cp = np.eye(4)
    s2 = 1.0 / np.sqrt(2.0)
    # cp[:3,:3] = np.array([
    #     [0.0, -s2, s2],
    #     [1.0, 0.0, 0.0],
    #     [0.0, s2, s2]
    # ])
    cp[:3,:3] = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    hfov = np.pi / 3.0
    dist = scene.scale / (2.0 * np.tan(hfov))
    cp[:3,3] = dist * np.array([0.0, 0.0, 0.0]) + centroid

    return cp

# print(camera_pose())

pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.5, znear=0.5, zfar=50.0)
# pc = pyrender.PerspectiveCamera(yfov=1.5, znear=0.01, zfar=100.0)
# print(pc.get_projection_matrix(width=1024, height=1024))
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
n = 0

tic = time.perf_counter()

for file in object_list:
    n += 1
    print(n)
    tic1 = time.perf_counter()
    input_mesh = trimesh.load(file)
    mesh = pyrender.Mesh.from_trimesh(input_mesh)
    nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
    scene.add_node(nm)
    tic2 = time.perf_counter()
    # print(f"Added object in {tic2 - tic1:0.4f} seconds")
    r = pyrender.OffscreenRenderer(viewport_width=1024, viewport_height=1024, point_size=1.0)
    tic3 = time.perf_counter()
    # print(f"Added r in {tic3 - tic2:0.4f} seconds")
    color, depth = r.render(scene)
    tic4 = time.perf_counter()
    # print(f"Rendered in {tic4 - tic3:0.4f} seconds")
    b,g,red = cv2.split(color)
    result = cv2.merge((red,g,b))
    cv2.imwrite(os.path.join(image_path, file.split(os.path.sep)[-1].replace('.obj','.png')), result)
    tic5 = time.perf_counter()
    # print(f"Saved image in {tic5 - tic4:0.4f} seconds")
    scene.remove_node(nm)
    tic6 = time.perf_counter()
    # print(f"Removed node in {tic6 - tic5:0.4f} seconds")
    r.delete()

#return timer
toc = time.perf_counter()
print(f"Generated images from objects in {toc - tic:0.4f} seconds")