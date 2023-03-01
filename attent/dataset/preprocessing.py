import os
import shutil
from time import time
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
from skimage.io import imread, imsave
from skimage import transform
import matplotlib.pyplot as plt 
# import dicom
from PIL import Image

dcm_dir = '/home/lc/Study/DataBase/CHAOS-MRI/39/T2SPIR/DICOM_anon/'#'/home/lc/Study/DataBase/multi-Atlas labeling beyond the cranial vault/Training/img/'
seg_dir = '/home/lc/Study/DataBase/CHAOS-MRI/39/T2SPIR/Ground/'#'/home/lc/Study/DataBase/multi-Atlas labeling beyond the cranial vault/Training/label/'
new_ct_dir = '/home/lc/Study/DataBase/CHAOS-MRI/image2/'
new_seg_dir = '/home/lc/Study/DataBase/CHAOS-MRI/label2/'

upper = 350#multi-A#200
lower = -upper#-200
expand_slice = 10  # 轴向上向外扩张的slice数量
size = 48  # 取样的slice数量
stride = 3  # 取样的步长
down_scale = 0.5
slice_thickness = 100

if not os.path.exists(new_ct_dir):
    os.mkdir(new_ct_dir)
    os.mkdir(new_seg_dir)
# os.mkdir(new_ct_dir)
# os.mkdir(new_seg_dir)
file_index = 0
test_index = 0
for dcm_file in os.listdir(dcm_dir):

    # 将CT和金标准入读内存
    ct = sitk.ReadImage(os.path.join(dcm_dir, dcm_file), sitk.sitkInt16)

    seg = sitk.ReadImage(os.path.join(seg_dir, dcm_file.replace('DICOM_anon', 'Ground').replace('dcm', 'png')), sitk.sitkInt16)

    ct_name=dcm_file.replace('.dcm', '')
    seg_name = ct_name#.replace('', 'label')
    # print(dcm_file)
    # seg = ct.resize(256,256)

    ct_array = sitk.GetArrayFromImage(ct)
    # print(ct_array.shape)


    seg_array = sitk.GetArrayFromImage(seg)
    seg_array1 = sitk.GetArrayFromImage(seg)
    seg_array2 = sitk.GetArrayFromImage(seg)
    seg_array3 = sitk.GetArrayFromImage(seg)
    seg_array4 = sitk.GetArrayFromImage(seg)

    print(seg_array.shape)

    new_ct_array = ct_array[0, :, :]
    new_seg_array = seg_array#[0, :, :]

    count_0=sum(new_seg_array!=0)   

    if sum(count_0)>0:

        new_ct_name =  ct_name + '.png'
        new_seg_name = seg_name+ '.png'
        
        rightKidney_label = seg_array1
        rightKidney_label[rightKidney_label!=120]=0
        rightKidney_label[rightKidney_label==120]=255
        if not os.path.exists(os.path.join(new_seg_dir, 'rightKidney_label')):
            os.mkdir(os.path.join(new_seg_dir, 'rightKidney_label'))
        imsave(new_seg_dir+'rightKidney_label/'+new_seg_name, rightKidney_label)

        liver_label = seg_array2
        liver_label[liver_label!=63]=0
        liver_label[liver_label==63]=255
        if not os.path.exists(os.path.join(new_seg_dir, 'liver_label')):
            os.mkdir(os.path.join(new_seg_dir, 'liver_label'))
        imsave(new_seg_dir+'liver_label/'+new_seg_name, liver_label)

        leftKidney_label = seg_array3
        leftKidney_label[leftKidney_label!=189]=0
        leftKidney_label[leftKidney_label==189]=255
        if not os.path.exists(os.path.join(new_seg_dir, 'leftKidney_label')):
            os.mkdir(os.path.join(new_seg_dir, 'leftKidney_label'))
        imsave(new_seg_dir+'leftKidney_label/'+new_seg_name, leftKidney_label)
        spleen_label = seg_array4
        spleen_label[spleen_label!=252]=0
        spleen_label[spleen_label==252]=255
        if not os.path.exists(os.path.join(new_seg_dir, 'spleen_label')):
            os.mkdir(os.path.join(new_seg_dir, 'spleen_label'))
        imsave(new_seg_dir+'spleen_label/'+new_seg_name, spleen_label)

        imsave(new_seg_dir+new_seg_name, new_seg_array)
        imsave(new_ct_dir+new_ct_name, new_ct_array)
    file_index = 0
