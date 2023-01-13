from regex import F
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
# import pdbkaleido
import scipy
# from three_view import *
import cv2
from turtle import color
import plotly.graph_objects as go
import plotly.express as px
from scipy import ndimage
from skimage.transform import resize

def get_cutbin(bin_res, wall_size, binlenth):
    '''
    to let each axis to replace same real lenth
    bin_res: bin_resolution
    wall_size: x and y axis repalce real lenth
    binlenth: data time resolution
    '''
    time_re = bin_res
    distance = binlenth * time_re * 3e8 /2
    cut_bin = int((distance-wall_size)*2 / 3e8 /time_re)
    return binlenth - cut_bin

def data_tranform(datapath , cut_bin, res=64):
    '''
    to transform data to same resolution in all axises
    data: lctresult_path
    cut_bin: need cut to make depth resolution same to x and y
    res: need resolution
    '''
    data = scio.loadmat(datapath) #x,y,z
    data = data['lct']
    data = data[:,:,:cut_bin]
    data = data / data.max()
    data = resize(data, (res,res,res))
    
    return data

def transform_matrix(u, v, w, a, b, c, t):
    '''
    u, v, w = 0, 1, 0           # unit vector of the ratate axis 
    a, b, c = 32, 0, 32         # one point of the axis 
    t = 45/180*np.pi
    '''
    mat = np.array([[u*u+(v*v+w*w)*np.cos(t), u*v*(1-np.cos(t))-w*np.sin(t), u*w*(1-np.cos(t))+v*np.sin(t), (a*(v*v+w*w)-u*(b*v+c*w))*(1-np.cos(t))+(b*w-c*v)*np.sin(t)],
                [u*v*(1-np.cos(t))+w*np.sin(t), v*v+(u*u+w*w)*np.cos(t),    v*w*(1-np.cos(t))-u*np.sin(t), (b*(u*u+w*w)-v*(a*u+c*w))*(1-np.cos(t))+(c*u-a*w)*np.sin(t)],
                [u*w*(1-np.cos(t))-v*np.sin(t), v*w*(1-np.cos(t))+u*np.sin(t), w*w+(u*u+v*v)*np.cos(t),    (c*(u*u+v*v)-w*(a*u+b*v))*(1-np.cos(t))+(a*v-b*u)*np.sin(t)],
                [0,                       0,                       0,                       1]])
    return mat

def rigid_transform_3D(A, B):
    A = A.transpose(1,0)
    B = B.transpose(1,0)
    assert len(A) == len(B)
    N = A.shape[0]
    mu_A = np.mean(A, axis=0)
    mu_B = np.mean(B, axis=0)

    AA = A - np.tile(mu_A, (N, 1))
    BB = B - np.tile(mu_B, (N, 1))
    # H = np.transpose(AA) * BB
    H = np.dot(np.transpose(AA) ,BB)

    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T , U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T , U.T)

    t = -np.dot(R, mu_A.T) + mu_B.T
    # R, T = R, t
    matrix = np.zeros((4,4))
    matrix[:3,:3] = R
    matrix[0:3,-1] = t
    matrix[-1,-1] = 1

    return matrix


def compute_normal(sample):
    point1_x = sample[0,0]
    point1_y = sample[1,0]
    point1_z = sample[2,0]
    point2_x = sample[0,1]
    point2_y = sample[1,1]
    point2_z = sample[2,1]
    point3_x = sample[0,2]
    point3_y = sample[1,2]
    point3_z = sample[2,2]
    na = (point2_y - point1_y)*(point3_z - point1_z) - (point2_z - point1_z)*(point3_y - point1_y)
    nb = (point2_z - point1_z)*(point3_x- point1_x) - (point2_x - point1_x)*(point3_z - point1_z)
    nc = (point2_x - point1_x)*(point3_y- point1_y) - (point2_y - point1_y)*(point3_x - point1_x)
    # vec_a = ((point2_x-point1_x), (point2_y-point1_y), (point2_z-point1_z))
    # vec_b = ((point3_x-point1_x), (point3_y-point1_y), (point3_z-point1_z))
    return np.array((na, nb, nc))

def getwall_transofrom(wallpath, reso = 64):
    walldata = scio.loadmat(wallpath)
    wall1_cood = walldata['parameters_galvo_system_coordinates']
    wall1_cood_new = np.zeros_like(wall1_cood)

    ##change wall axis 
    # fro 0113 eral data
    # wall1_cood_new[0] = wall1_cood[0]
    # wall1_cood_new[1] = -wall1_cood[2]
    # wall1_cood_new[2] = wall1_cood[1]

    #for cvpr turtul data
    wall1_cood_new[0] = wall1_cood[0]
    wall1_cood_new[1] = wall1_cood[1]
    wall1_cood_new[2] = wall1_cood[2]
    point1_x = wall1_cood_new[0,0]
    point1_y = wall1_cood_new[1,0]
    point1_z = wall1_cood_new[2,0]
    point2_x = wall1_cood_new[0,63]
    point2_y = wall1_cood_new[1,63]
    point2_z = wall1_cood_new[2,63]
    wall_size = np.sqrt((point1_x - point2_x)**2 + (point1_y - point2_y) **2 \
    + (point1_z - point2_z)**2)

    wall1_sample = np.zeros((3,3))
    wall1_sample[:,0] = wall1_cood_new[:,0]
    wall1_sample[:,1] = wall1_cood_new[:,63]
    wall1_sample[:,2] = wall1_cood_new[:,-1]
    wall_l_normal = compute_normal(wall1_sample)

    camera_z, camera_x = np.meshgrid(np.linspace(wall_size/2, -wall_size/2, reso), np.linspace(wall_size/2, -wall_size/2, reso), indexing='ij') 
    camera_y = np.ones_like(camera_x)
    coords = np.stack((camera_x,camera_y,camera_z), 0)
    coords = coords.reshape(3,-1)
    wall1_matrix = rigid_transform_3D(coords, wall1_cood_new)

    return wall1_matrix, wall_l_normal


def mask(data, og_volume, threshold=0.1, dilateflag=False):
    front = data.max(axis=2)
    front_index = data.argmax(axis=2)
    index_min = front_index.min()
    threshold_f = front.max()*threshold
    if dilateflag==True:
        kernel = np.ones((10, 10), np.uint8)    
        front = cv2.dilate(front, kernel)
    front[front<threshold_f] = 0
    index_front = np.argwhere(front==0)
    og_volume[index_front[:,0],index_front[:,1],:] = 0
    og_volume[:,:,:index_min] = 0
    return og_volume

if __name__ == '__main__':
    u, v, w = 1, 0, 0           # unit vector of the ratate axis 
    a, b, c = 0, 31.5, 31.5   # one point of the axis mid resolutin
    t0 = -120/180*np.pi
    t1 = -240/180*np.pi
    res = 64
    wall_is_right = True ##wall is vertical to the ground 

    datadir = 'D:/Desktop/xiaopeng/dlct_re/'
    data0_name = '0.mat'
    data1_name = '120.mat'
    data2_name = '240.mat'
    res_name = 'dlct_re.mat'
    bin_re = 0.01/3e8
    wall_size = 2
    binlenth = 512
    px_vis = True   #vis for plotly
    three_view = True
    use_forpretrain = True ## need mask 

    if wall_is_right:
        mat0 = transform_matrix(u, v, w, a, b, c, t0)
        mat1 = transform_matrix(u, v, w, a, b, c, t1)
        cutbin = get_cutbin(bin_re,wall_size, binlenth)
        frontpath = datadir + data0_name
        front = data_tranform(frontpath,cutbin, res)
        if use_forpretrain:
            og_volume = np.ones((res,res,res))
            volume1 = mask(front, og_volume, threshold=0.1)
        tailpath = datadir + data1_name
        tail = data_tranform(tailpath,cutbin, res)
        if use_forpretrain:
            og_volume = np.ones((res,res,res))
            volume2 = mask(tail, og_volume, threshold=0.05, dilateflag=True)
            volume2 = ndimage.affine_transform(volume2, mat0)
        tail = ndimage.affine_transform(tail,mat0)
        earpath = datadir + data2_name
        ear = data_tranform(earpath,cutbin, res)
        if use_forpretrain:
            og_volume = np.ones((res,res,res))
            volume3 = mask(ear, og_volume, threshold=0.05, dilateflag=True)
            volume3 = ndimage.affine_transform(volume3, mat1)
        ear = ndimage.affine_transform(ear,mat1)
        values = front + ear + tail
        if use_forpretrain:
            values = volume1 + volume2 +volume3

    if wall_is_right == False:
        '''
        need real data_trainsient to get transofrm matrix
        '''
        wall1path = ''
        wall2path = ''
        mat0, wall_norm = getwall_transofrom(wall1path)
        mat1, wall_norm = getwall_transofrom(wall2path)
        cutbin = get_cutbin(bin_re,wall_size, binlenth)
        frontpath = datadir + data0_name
        front = data_tranform(frontpath,cutbin, res)
        if use_forpretrain:
            og_volume = np.ones((res,res,res))
            volume1 = mask(front, og_volume, threshold=0.1)
        tailpath = datadir + data1_name
        tail = data_tranform(tailpath,cutbin, res)
        if use_forpretrain:
            og_volume = np.ones((res,res,res))
            volume2 = mask(tail, og_volume, threshold=0.05, dilateflag=True)
            volume2 = ndimage.affine_transform(volume2, mat0)
        tail = ndimage.affine_transform(tail,mat0)
        earpath = datadir + data2_name
        ear = data_tranform(earpath,cutbin, res)
        if use_forpretrain:
            og_volume = np.ones((res,res,res))
            volume3 = mask(ear, og_volume, threshold=0.05, dilateflag=True)
            volume3 = ndimage.affine_transform(volume3, mat1)
        ear = ndimage.affine_transform(ear,mat1)
        values = front + ear + tail

        pass

    if px_vis:
        X, Y, Z = np.mgrid[0:64:values.shape[0]*1j, 0:64:values.shape[1]*1j, 0:64:values.shape[2]*1j]

        # pdb.set_trace()
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomin=values.min(),
            isomax=values.max(),
            opacity=0.8,
            opacityscale="max",
        
            colorscale='hot',
            surface_count=10,
        ))
        
        bgcolor = px.colors.sequential.gray[0]
        fig.update_layout(
        
            scene=dict(
            xaxis=dict(tickmode='auto', backgroundcolor=bgcolor, nticks=1, range=[0, 64], showgrid=True, showline=True, color='white', mirror=True, showticklabels=False, showaxeslabels=False, visible=True, zeroline=False),
            yaxis=dict(tickmode='auto', backgroundcolor=bgcolor, nticks=1, range=[0, 64], showgrid=True, showline=True, color='white', mirror=True, showticklabels=False, showaxeslabels=False, visible=True, zeroline=False),
            zaxis=dict(tickmode='auto', backgroundcolor=bgcolor, nticks=1, range=[0, 64], showgrid=True, showline=True, color='white', mirror=True, showticklabels=False, showaxeslabels=False, visible=True, zeroline=False),
            xaxis_title='',
            yaxis_title='',
            zaxis_title=''),
              
            scene_camera=dict( # cubes
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=100, y=100, z=100),
                projection=dict(type='orthographic')
            )
            )
        
        fig.update_traces(showscale=True)
        # fig.write_image('bunny_side.png')
        fig.show()

    ## save values
    svname = ''
    scio.savemat(svname,{"data":values})
        