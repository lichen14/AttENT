from pathlib import Path
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils import data
import cv2

class BaseDataset(data.Dataset):
    def __init__(self, root, list_path, set_,
                 max_iters, image_size, labels_size, mean):
        self.root = Path(root)
        self.set = set_
        self.list_path = list_path.format(self.set)
        self.image_size = (256,256)#image_size
        # self.transform = transforms.Compose(transforms_)
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean = mean
        with open(self.list_path) as f:

            self.img_ids = [i_id.strip() for i_id in f]
        if max_iters is not None:
            self.img_ids = self.img_ids#* int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # print(len(self.img_ids))
        for name in self.img_ids:
            img_file, label_file,label_file_liver,label_file_rightK,label_file_leftK,label_file_spleen = self.get_metadata(name)
            self.files.append((img_file, label_file,label_file_liver,label_file_rightK,label_file_leftK,label_file_spleen, name))
        # print(len(self.files))
    def get_metadata(self, name):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))

    def get_image(self, file):

        return _load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_CHAOS_image(self, file):
        return _load_CHAOS_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_CHAOS_labels(self, file):
        return _load_CHAOS_img(file, self.labels_size, Image.NEAREST, rgb=False)
    def get_labels(self, file):
        return _load_label(file, self.labels_size, Image.NEAREST, rgb=False)

    def get_robert_edges(self, file):

        return custom_suanzi(file)
    def get_sobel_edges(self, file):
        return sobel_suanzi(file)
    def get_Laplace_edges(self, file):
        return Laplace_suanzi(file)

def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    # print(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    # print('IMG SHAPE2',np.array(img).shape)
    return np.asarray(img, np.float32)#img#

def _load_label(file, size, interpolation, rgb):
    img = Image.open(file)

    # print(size)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    # img.show()
    img_array = np.asarray(img, np.float32)
    # print('label SHAPE2',img_array.shape)
    # print('0',img_array[125,125,0],'1',img_array[125,125,1],'2',img_array[125,125,2])
    return np.asarray(img, np.float32)

def _load_CHAOS_img(file, size, interpolation, rgb):
    img = Image.open(file)
    # print(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    img_array = np.asarray(img, np.float32)
    # print('CHAOS-label SHAPE2',img_array.shape)
    # print('0',img_array[125,125,0],'1',img_array[125,125,1],'2',img_array[125,125,2])
    return img_array[:,:]#np.asarray(img, np.float32)

# robert 算子[[-1,-1],[1,1]]
def robert_suanzi(img):
    r, c = img.shape
    r_sunnzi = [[-1,-1],[1,1]]
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                imgChild = img[x:x+2, y:y+2]
                list_robert = r_sunnzi*imgChild
                img[x, y] = abs(list_robert.sum())   # 求和加绝对值
    return img

# # sobel算子的实现
def sobel_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])   # X方向
    s_suanziY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    for i in range(r-2):
        for j in range(c-2):
            new_imageX[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziX))
            new_imageY[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziY))
            new_image[i+1, j+1] = (new_imageX[i+1, j+1]*new_imageX[i+1,j+1] + new_imageY[i+1, j+1]*new_imageY[i+1,j+1])**0.5
    # return np.uint8(new_imageX)
    # return np.uint8(new_imageY)
    return np.uint8(new_image) # 无方向算子处理的图像

# Laplace算子
# 常用的Laplace算子模板 [[0,1,0],[1,-4,1],[0,1,0]]  [[1,1,1],[1,-8,1],[1,1,1]]
def Laplace_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    L_sunnzi = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    # L_sunnzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    for i in range(r-2):
        for j in range(c-2):
            new_image[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * L_sunnzi))
    return np.uint8(new_image)

def custom_suanzi(img):
    vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
    # 定义横向过滤器
    horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]
    # 得到图片的维数
    n,m = img.shape
    # 初始化边缘图像
    edges_img = img.copy()
    # 循环遍历图片的全部像素
    for row in range(3, n-2):
        for col in range(3, m-2):
            # 在当前位置创建一个 3x3 的小方框
            local_pixels = img[row-1:row+2, col-1:col+2]

            # 应用纵向过滤器
            vertical_transformed_pixels = vertical_filter*local_pixels
            # print(vertical_transformed_pixels.shape)
            # 计算纵向边缘得分
            vertical_score = vertical_transformed_pixels.sum()/4

            # 应用横向过滤器
            horizontal_transformed_pixels = horizontal_filter*local_pixels
            # 计算横向边缘得分
            horizontal_score = horizontal_transformed_pixels.sum()/4

            # 将纵向得分与横向得分结合，得到此像素总的边缘得分
            edge_score = (vertical_score**2 + horizontal_score**2)**.5
            # print(edge_score.shape,edges_img.shape)
            # 将边缘得分插入边缘图像中
            edges_img[row, col] =edge_score*3
    # 对边缘图像中的得分值归一化，防止得分超出 0-1 的范围
    # edges_img = edges_img/edges_img.max()
    return edges_img
# def robert_detection(path):
#     # print(path)
#     img = Image.open(path)
#     img_array = np.asarray(img, np.float32)
#     # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     # cv2.imshow('image', img)
#     out_robert = robert_suanzi(img_array)
#     # cv2.imshow('out_robert_image', out_robert)
#     return out_robert
#     # # robers算子
# def sobel_detection(path):
#     img = Image.open(path)
#     img_array = np.asarray(img, np.float32)
#     out_sobel = sobel_suanzi(img_array)
#     # cv2.imshow('out_sobel_image', out_sobel)
#     return out_sobel
#     # Laplace算子
#
# def Laplace_detection(path):
#     img = Image.open(path)
#     img_array = np.asarray(img, np.float32)
#     out_laplace = Laplace_suanzi(img_array)
#     # cv2.imshow('out_laplace_image', out_laplace)
#     return out_laplace
