#1.use boundary mat to get obj
#2.need boundary obe to get sdf
from skimage import measure
import scipy.io as sio
import torch
import numpy as np
import matplotlib.pyplot as plt
import mcubes
import trimesh
from skimage.transform import resize
import mesh_to_sdf 

def saveobj2(output, result_path='./results/', i=0, threshold=0.1,name = "bunny_test"):
    volume = output
    vertices, triangles = mcubes.marching_cubes(volume, threshold * volume.max())
    mesh = trimesh.Trimesh(vertices, triangles)
    trimesh.repair.fill_holes(mesh)
    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)
    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    print('down')
    return mesh

def transform2net(data, res=256):
    data = data.reshape(res,res,res)
    sdf = resize(data,(res,res,res))
    sdf_new = np.flip(sdf, axis=2)
    sdf_new2 =  sdf_new.transpose(1,0,2)
    sdf_new3 = np.flip(sdf_new2, axis=1)
    return sdf_new3

if __name__ == '__main__':
    datapath = '' # boundary mat path 
    data = sio.loadmat(datapath)['data']
    new_data = transform2net(data)
    boundary_obj = saveobj2(data) ## p
    x, y, z = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256), np.linspace(-1, 1, 256), indexing='ij')
    coords = np.stack((x, y, z), 0).reshape(3, -1)
    coords = np.transpose(coords, [1, 0])
    sdf = mesh_to_sdf.mesh_to_sdf(boundary_obj, coords,surface_point_method='scan', sign_method='normal')#surface_point_method 'scan'
    # gt_sdf_all = sio.loadmat("./bunny_lcttest_sdf.mat")['ft_sdf']  sample  bounding_radius=1
    gt_sdf_all = {}
    gt_sdf_all['ft_sdf'] = sdf
    svname = './results/density_boundary_serapis_sdf.mat'
    sio.savemat(f'{svname}',gt_sdf_all )
    
