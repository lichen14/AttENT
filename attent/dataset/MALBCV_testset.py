import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
from dataset.base_dataset import BaseDataset


class MALBCV_testSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(256, 256), mean=(128, 128, 128), labels_size=None):
        super().__init__(root, list_path, set,  crop_size, labels_size, labels_size,mean)

        # map to cityscape's ids
        self.class_names = ['background','liver','rightKidney','leftKidney','spleen']
        self.id_to_trainid = {42: 4, 85: 2, 127: 3, 255: 1}

    def get_metadata(self, name):
        img_file = self.root / 'test' /'B-translated10' / name#
        label_file = self.root / 'test' /'B_label' / name
        label_file_liver = self.root/'test' /'B_label'/'liver_label'/ name#/ 'test'/'liver_label / name
        label_file_rightK = self.root/'test' /'B_label'/'rightKidney_label'/ name#/ 'test'/'liver_label / name
        label_file_leftK = self.root/'test' /'B_label'/'leftKidney_label'/ name#/ 'test'/'liver_label / name
        label_file_spleen = self.root/'test' /'B_label'/'spleen_label'/ name#/ 'test'/'liver_label / name
        # img_file = self.root / 'Train'/'image' / name
        # label_file = self.root / 'Train'/'mask' / name
        return img_file, label_file,label_file_liver,label_file_rightK,label_file_leftK,label_file_spleen

    def __getitem__(self, index):
        img_file, label_file,label_file_liver,label_file_rightK,label_file_leftK,label_file_spleen, name = self.files[index]
        # print(img_file, label_file, name )
        image = self.get_image(img_file)
        label = self.get_labels(label_file)

        label_rightK = self.get_labels(label_file_rightK)
        label_leftK = self.get_labels(label_file_leftK)
        label_liver = self.get_labels(label_file_liver)
        label_spleen = self.get_labels(label_file_spleen)
        # print('label.shape:',label.shape)#(256, 256, 3)
        # re-assign labels to match the format of Cityscapes
        label_copy = np.zeros(label.shape, dtype=np.float32)
        # for k, v in self.id_to_trainid.items():
        #     label_copy[label == k] = v
        label_copy[label_liver >250] = 1
        label_copy[label_rightK >250] = 2
        label_copy[label_leftK >250] = 3
        label_copy[label_spleen >250] = 4
        
        image = self.preprocess(image)
        # print('images:',image.copy().shape,'target_labels',label_copy.copy().shape)
        return image.copy(), label_copy.copy(), np.array(image.shape), name
