import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize

def gaussian_create():
    sigma1 = sigma2 = 1
    gaussian_sum = 0
    g = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            g[i, j] = math.exp(-1 / 2 * (np.square(i - 1) / np.square(sigma1)
                                         + (np.square(j - 1) / np.square(sigma2)))) / (
                              2 * math.pi * sigma1 * sigma2)
            gaussian_sum = gaussian_sum + g[i, j]
    g = g / gaussian_sum  # 归一化
    return g



# def gray_fuc(rgb):
#     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gaussian_blur(gray_img, g):
    gray_img = np.pad(gray_img, ((1, 1), (1, 1)), constant_values=0)  # 填充
    h, w = gray_img.shape
    new_gray_img = np.zeros([h - 2, w - 2])
    for i in range(h - 2):
        for j in range(w - 2):
            new_gray_img[i, j] = np.sum(gray_img[i:i + 3, j:j + 3] * g)
    return new_gray_img


# 求高斯偏导
def partial_derivative(new_gray_img):
    new_gray_img = np.pad(new_gray_img, ((0, 1), (0, 1)), constant_values=0)  # 填充
    h, w = new_gray_img.shape
    dx_gray = np.zeros([h - 1, w - 1])  # 用来存储x方向偏导
    dy_gray = np.zeros([h - 1, w - 1])  # 用来存储y方向偏导
    df_gray = np.zeros([h - 1, w - 1])  # 用来存储梯度强度
    for i in range(h - 1):
        for j in range(w - 1):
            dx_gray[i, j] = new_gray_img[i, j + 1] - new_gray_img[i, j]
            dy_gray[i, j] = new_gray_img[i + 1, j] - new_gray_img[i, j]
            df_gray[i, j] = np.sqrt(np.square(dx_gray[i, j]) + np.square(dy_gray[i, j]))
    return dx_gray, dy_gray, df_gray


# 非极大值抑制
def non_maximum_suppression(dx_gray, dy_gray, df_gray):
    df_gray = np.pad(df_gray, ((1, 1), (1, 1)), constant_values=0)  # 填充
    h, w = df_gray.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if df_gray[i, j] != 0:
                gx = math.fabs(dx_gray[i - 1, j - 1])
                gy = math.fabs(dy_gray[i - 1, j - 1])
                if gx > gy:
                    weight = gy / gx
                    grad1 = df_gray[i + 1, j]
                    grad2 = df_gray[i - 1, j]
                    if gx * gy > 0:
                        grad3 = df_gray[i + 1, j + 1]
                        grad4 = df_gray[i - 1, j - 1]
                    else:
                        grad3 = df_gray[i + 1, j - 1]
                        grad4 = df_gray[i - 1, j + 1]
                else:
                    weight = gx / gy
                    grad1 = df_gray[i, j + 1]
                    grad2 = df_gray[i, j - 1]
                    if gx * gy > 0:
                        grad3 = df_gray[i + 1, j + 1]
                        grad4 = df_gray[i - 1, j - 1]
                    else:
                        grad3 = df_gray[i + 1, j - 1]
                        grad4 = df_gray[i - 1, j + 1]
                t1 = weight * grad1 + (1 - weight) * grad3
                t2 = weight * grad2 + (1 - weight) * grad4
                if df_gray[i, j] > t1 and df_gray[i, j] > t2:
                    df_gray[i, j] = df_gray[i, j]
                else:
                    df_gray[i, j] = 0
    return df_gray


# 双阈值过滤
def double_threshold(df_gray, low, high):
    h, w = df_gray.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if df_gray[i, j] < low:
                df_gray[i, j] = 0
            elif df_gray[i, j] > high:
                df_gray[i, j] = 1
            elif (df_gray[i, j - 1] > high) or (df_gray[i - 1, j - 1] > high) or (
                    df_gray[i + 1, j - 1] > high) or (df_gray[i - 1, j] > high) or (df_gray[i + 1, j] > high) or (
                    df_gray[i - 1, j + 1] > high) or (df_gray[i, j + 1] > high) or (df_gray[i + 1, j + 1] > high):
                df_gray[i, j] = 1
            else:
                df_gray[i, j] = 0
    return df_gray


def canny(img_og):
    ksize = (10, 10)
    img = cv2.blur(img_og , ksize)
    img = Image.fromarray(np.uint8(img))
    gray = np.array(img)
    gray[gray>0] = 255
    gaussian = gaussian_create()
    new_gray = gaussian_blur(gray, gaussian)
    d = partial_derivative(new_gray)
    dx = d[0]
    dy = d[1]
    df = d[2]
    new_df = non_maximum_suppression(dx, dy, df)
    low_threshold = 0.15 * np.max(new_df)
    high_threshold = 0.2 * np.max(new_df)
    result = double_threshold(new_df, low_threshold, high_threshold)
    ksize2 = (5,5)
    result = cv2.blur(result, ksize2)
    result = resize(result,(256,256))
    return result

if __name__ == '__main__':
    import scipy.io as sio
    import cv2
    density = sio.loadmat(r'density_dilate_serapis.mat')['data']
    x,y,z = density.shape
    density_boundary = np.zeros_like(density)
    for i in range(z):
        img = density[:,:,i]
        result = canny(img)
        density_boundary[:,:,i] = result
    print(density_boundary.shape)
    sio.savemat('density_boundary_serapis.mat',{'data':density_boundary})


