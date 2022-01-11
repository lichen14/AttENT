import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
from dataset.base_dataset import BaseDataset


class CHAOS_testSet(BaseDataset):
    def __init__(self, root, list_path, set='test',
                 max_iters=None, crop_size=(256, 256), mean=(128, 128, 128), labels_size=None):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)

        # map to cityscape's ids
        self.class_names = ['background','liver','rightKidney','leftKidney','spleen']
        self.id_to_trainid = {63: 1, 126: 2, 189: 3, 252: 4}

    def get_metadata(self, name):
        img_file = self.root/ 'test' /'A_image'/ name#/ 'test'/'translated_image' / name
        label_file = self.root/'test' /'A_label'/ name#/ 'test'/'liver_label / name
        label_file_liver = self.root/'test' /'A_label'/'liver_label'/ name#/ 'test'/'liver_label / name
        label_file_rightK = self.root/'test' /'A_label'/'rightKidney_label'/ name#/ 'test'/'liver_label / name
        label_file_leftK = self.root/'test' /'A_label'/'leftKidney_label'/ name#/ 'test'/'liver_label / name
        label_file_spleen = self.root/'test' /'A_label'/'spleen_label'/ name#/ 'test'/'liver_label / name
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
        # plt.subplot(221)
        # plt.imshow(image.astype(np.uint8))
        # # plt.show()
        # plt.subplot(222)
        # plt.imshow(label)
        # plt.subplot(223)
        # plt.imshow(label_copy)
        # plt.show()
        image = self.preprocess(image)
        #images: (3, 256, 256) label_copy (256, 256, 3)
        return image.copy(), label_copy.copy(), np.array(image.shape), name
