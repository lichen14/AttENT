import numpy as np
import torch
import torch.nn as nn
import medpy.metric.binary as mmb
from utils.loss import cross_entropy_2d
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # plt 用于显示图片
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
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

def robert_detection(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    out_robert = robert_suanzi(img)
    cv2.imshow('out_robert_image', out_robert)
    return out_robert
    # # robers算子
def sobel_detection(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    out_sobel = sobel_suanzi(img)
    cv2.imshow('out_sobel_image', out_sobel)
    return out_sobel
    # Laplace算子

def Laplace_detection(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    out_laplace = Laplace_suanzi(img)
    cv2.imshow('out_laplace_image', out_laplace)
    return out_laplace

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())   #造一个Predict大小的tensor
    y_truth_tensor.fill_(y_label)                       #用label填充它
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device()) #将其送入Predict对应的device里
    # print('y_truth_tensor,',y_truth_tensor.shape)
    # print('y_pred,',y_pred.shape)
    # t=y_pred[0,0,:,:]
    # # print(t.shape)
    # tt = np.argmax(F.softmax(predict).cpu().data[0].numpy().transpose(1, 2, 0),axis=2)
    # # print(tt.shape)   #(256.256)
    # # tt=np.zeros((256,256,3))
    # # tt[:,:,0]=t[0,:,:]
    # # # tt[:,:,1]=t[0,:,:]
    # # # tt[:,:,2]=t[1,:,:]
    # # print(tt[0,100,100])
    # # m=target[1,:,:,:]
    # # mm=np.zeros((256,256))
    # mm=y_label[0,:,:]
    # ##print(np.maximum(mm, -1))
    # # # mm[:,:,1]=target[1,:,:]
    # # # mm[:,:,2]=target[2,:,:]
    # # # t = t.transpose((1,2,0))
    # plt.subplot(121)
    # plt.imshow(tt)
    # # plt.show()
    # plt.subplot(122)
    # plt.imshow(mm.cpu().data)#, cmap='Greys_r') # 显示图片
    # # plt.axis('off') # 不显示坐标轴
    # plt.show()
    # tmp1 = nn.Sigmoid()(y_pred)
    # loss1= nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)
    # loss2 = nn.BCELoss()(tmp1, y_truth_tensor)
    # print('tmp1 and y_truth_tensor :',tmp1,y_truth_tensor)
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)       #[1, 1, 8, 8],[1, 1, 8, 8]

def bce_loss2(y_pred, y_label):
    # y_truth_tensor = torch.FloatTensor(y_pred.size())   #造一个Predict大小的tensor
    # y_truth_tensor.fill_(y_label)                       #用label填充它
    # y_truth_tensor = y_truth_tensor.to(y_pred.get_device()) #将其送入Predict对应的device里
    # print('y_truth_tensor,',y_truth_tensor.shape)
    # print('y_pred,',y_pred.shape)
    # t=predict[0,0,:,:]
    # # print(t.shape)
    # tt = np.argmax(F.softmax(predict).cpu().data[0].numpy().transpose(1, 2, 0),axis=2)
    # # print(tt.shape)   #(256.256)
    # # tt=np.zeros((256,256,3))
    # # tt[:,:,0]=t[0,:,:]
    # # # tt[:,:,1]=t[0,:,:]
    # # # tt[:,:,2]=t[1,:,:]
    # # print(tt[0,100,100])
    # # m=target[1,:,:,:]
    # # mm=np.zeros((256,256))
    # mm=target[0,:,:]
    # ##print(np.maximum(mm, -1))
    # # # mm[:,:,1]=target[1,:,:]
    # # # mm[:,:,2]=target[2,:,:]
    # # # t = t.transpose((1,2,0))
    # plt.subplot(121)
    # plt.imshow(tt)
    # # plt.show()
    # plt.subplot(122)
    # plt.imshow(mm.cpu().data)#, cmap='Greys_r') # 显示图片
    # # plt.axis('off') # 不显示坐标轴
    # plt.show()
    # tmp1 = nn.Sigmoid()(y_pred)
    # loss1= nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)
    # loss2 = nn.BCELoss()(tmp1, y_truth_tensor)
    # print('tmp1 and y_truth_tensor :',tmp1,y_truth_tensor)
    return nn.BCEWithLogitsLoss()(y_pred, y_label)       #[1, 1, 8, 8],[1, 1, 8, 8]

def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # print(label)
    label = label.long().to(device)
    return cross_entropy_2d(pred, label)

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    公式（2）
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def prob_2_entropy_custom(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    公式（2）
    """
    n, c, h, w = prob.size()
    # tmp = torch.zeros(prob.shape)#torch.zeros_like(prob)
    # label1 = torch.zeros(prob.shape)#tmp[prob==1]
    # label2 = torch.zeros(prob.shape)#tmp[prob==2]
    # label3 = torch.zeros(prob.shape)#tmp[prob==3]
    # label4 = torch.zeros(prob.shape)#tmp[prob==4]
    #
    # label1[prob==1] =1
    # label2[prob==2] =2
    # label3[prob==3] =3
    # label4[prob==4] =4

    label1_entropy = -torch.mul(prob[:,1,:,:], torch.log2(prob[:,1,:,:] + 1e-30)) / np.log2(c)
    label2_entropy = -torch.mul(prob[:,2,:,:], torch.log2(prob[:,2,:,:] + 1e-30)) / np.log2(c)
    label3_entropy = -torch.mul(prob[:,3,:,:], torch.log2(prob[:,3,:,:] + 1e-30)) / np.log2(c)
    label4_entropy = -torch.mul(prob[:,4,:,:], torch.log2(prob[:,4,:,:] + 1e-30)) / np.log2(c)
    # plt.subplot(221)
    # plt.imshow(label1_entropy[0,:,:].data.cpu())
    # plt.subplot(222)
    # plt.imshow(label2_entropy[0,:,:].data.cpu())
    # plt.subplot(223)
    # plt.imshow(label3_entropy[0,:,:].data.cpu())
    # plt.subplot(224)
    # plt.imshow(label4_entropy[0,:,:].data.cpu())
    # plt.show()
    result = label1_entropy + label2_entropy + label3_entropy + label4_entropy
    return result#-torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def discrepancy_calc(v1, v2):
    """
    dis_loss for two different classifiers
    input : v1,v2
    output : discrepancy
    """
    assert v1.dim() == 4
    assert v2.dim() == 4
    n, c, h, w = v1.size()
    inner = torch.mul(v1, v2)   #pixel-wise point multiply
    v1 = v1.permute(2, 3, 1, 0)
    v2 = v2.permute(2, 3, 0, 1) #transpose
    mul = v1.matmul(v2)         # matrix multiply
    mul = mul.permute(2, 3, 0, 1)
    dis = torch.sum(mul) - torch.sum(inner)
    dis = dis / (h * w)
    return dis


def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)


def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D)


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    公式（2）
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def fast_hist(a, b, n): #label.flatten(), output.flatten(), cfg.NUM_CLASSES
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # print(output.shape)
    # target = target /255
    # output = output* 255
    iou_s = []


    target_ = target > 0
    output_ = output > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    # print(intersection,union)
    return (intersection + smooth) / (union + smooth) #np.mean(iou_s)#

def per_class_iou_score(output, target,ious,num_classes):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # print(output.shape)
    # target = target /255
    # output = output* 255

    for k in range(1,num_classes):
        pred_test_data_tr = output.copy()
        pred_gt_data_tr = target.copy()
        label_copy = np.zeros(pred_test_data_tr.shape, dtype=np.float32)
        gt_copy = np.zeros(pred_gt_data_tr.shape, dtype=np.float32)
        label_copy[pred_test_data_tr==k] =1
        gt_copy[pred_gt_data_tr==k] =1
        target_ = gt_copy > 0
        output_ = label_copy > 0.5
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        # if iou ==1 :
        #     # print('111111111')
        #     ious.append(np.nan)
        # else:
        ious.append(iou)
    # print(intersection,union)
    return ious #np.mean(iou_s)#

def per_class_dice(output, target,dices,num_classes):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # print(output.shape)
    # target = target /255
    # output = output* 255

    for k in range(1,num_classes):
        pred_test_data_tr = output.copy()
        pred_gt_data_tr = target.copy()
        # label_copy = np.zeros(pred_test_data_tr.shape, dtype=np.float32)
        # gt_copy = np.zeros(pred_gt_data_tr.shape, dtype=np.float32)
        pred_test_data_tr[pred_test_data_tr!=k] =0
        pred_gt_data_tr[pred_gt_data_tr!=k] =0
        dice1=mmb.dc(pred_test_data_tr, pred_gt_data_tr)

        target_ = pred_gt_data_tr > 0
        output_ = pred_test_data_tr > 0.5
        intersection = (output_ * target_).sum()
        # union = (output_ | target_).sum()

        dice2=(2. * intersection + smooth) / \
            (output_.sum() + target_.sum() + smooth)
        # if dice ==1 :
        #     # print('111111111')
        #     dices.append(np.nan)
        # else:
        # print('dice from medpy:',dice1,'dice from my:',dice2)
        dices.append(dice2)
    # print(intersection,union)
    # intersection = torch.sum((output_ + target_)==2)#(output_ * target_).sum()
    # print(intersection)
    # print(output.sum())
    # print(target.sum())
    return dices #np.mean(iou_s)#

def dice_coef(output, target):
    smooth = 1e-5
    # print(output_[10,200,200])
    # # print(target.shape)
    # print(target_[10,200,200])
    # output= torch.from_numpy(output)
    # print(target)
    # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    # target = target.view(-1).data.cpu().numpy()
    # target = target.astype('float32') * 255
    # output = output.astype('float32') * 255
    # if torch.is_tensor(output):
    #     output = torch.sigmoid(output).data.cpu().numpy()
    # if torch.is_tensor(target):
    #     target = target.data.cpu().numpy()

    #
    output_ = output > 0.5
    target_ = target > 0
    #
    # t=output[0,10,:,:].data.cpu().numpy()
    # # print(t.shape)
    # # tt=np.zeros((256,256,3))
    # # tt[:,:,0]=t[0,:,:]
    # # # tt[:,:,1]=t[0,:,:]
    # # # tt[:,:,2]=t[1,:,:]
    # # print(tt[0,100,100])
    # # # m=target[1,:,:,:]
    # # mm=np.zeros((512,512))
    # mm=target[0,10,:,:].data.cpu().numpy()
    # # # mm[:,:,1]=target[1,:,:]
    # # # mm[:,:,2]=target[2,:,:]
    # # # t = t.transpose((1,2,0))
    # # # print(tt.shape)
    # plt.subplot(121)
    # plt.imshow(t) # 显示图片
    # plt.subplot(122)
    # plt.imshow(mm, cmap='Greys_r') # 显示图片
    # # plt.axis('off') # 不显示坐标轴
    # plt.show()

    intersection = (output_ * target_).sum()
    # intersection = torch.sum((output_ + target_)==2)#(output_ * target_).sum()
    # print(intersection)
    # print(output.sum())
    # print(target.sum())
    return (2. * intersection + smooth) / \
        (output_.sum() + target_.sum() + smooth)
    #return 1-((2. * intersection + smooth) / \
    #    (torch.sum(output_) + torch.sum(target_) + smooth))

def dice_edge(output, target):
    smooth = 1e-5
    #vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
    # 定义横向过滤器
    #horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]
    # 得到图片的维数
    channel,n,m = output.shape
    # 初始化边缘图像
    edges_img = np.zeros(output.shape)#output.copy()
    # 循环遍历图片的全部像素
    for c in range(channel):
        for row in range(2, n):
            for col in range(m):
                # 在当前位置创建一个 3x3 的小方框
                previous_pixel = output[c,row-1, col]
                local_pixel = output[c,row, col]
                if local_pixel != previous_pixel:
                    edges_img[c,row, col]=1
                # local_pixels = output[c,row-1:row+2, col-1:col+2]

                # # 应用纵向过滤器
                # vertical_transformed_pixels = vertical_filter*local_pixels
                # # print(vertical_transformed_pixels.shape)
                # # 计算纵向边缘得分
                # vertical_score = vertical_transformed_pixels.sum()/4

                # # 应用横向过滤器
                # horizontal_transformed_pixels = horizontal_filter*local_pixels
                # # 计算横向边缘得分
                # horizontal_score = horizontal_transformed_pixels.sum()/4

                # # 将纵向得分与横向得分结合，得到此像素总的边缘得分
                # edge_score = (vertical_score**2 + horizontal_score**2)**.5
                # # print(edge_score.shape,edges_img.shape)
                # # 将边缘得分插入边缘图像中
                # edges_img[row, col] =edge_score*3

    output_ = torch.from_numpy(edges_img) > 0.5
    target_ = target > 0

    intersection = (output_ * target_).sum()
    return (2. * intersection + smooth) / \
        (output_.sum() + target_.sum() + smooth)
    #return 1-((2. * intersection + smooth) / \
    #    (torch.sum(output_) + torch.sum(target_) + smooth))

def edge_detection(output):
    #vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
    # 定义横向过滤器
    #horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]
    # 得到图片的维数
    output = output.transpose(0,1)
    n,c,m = output.shape
    # 初始化边缘图像
    edges_img = torch.zeros(output.shape)#output.copy()
    # 循环遍历图片的全部像素
    for r in range(2,n):
        mask = (output[r,:,:]!=output[r-1,:,:])
        #print(mask)
        edges_img[r,mask]=1
    # for c in range(channel):
    #     for row in range(2, n):
    #         for col in range(m):
    #             # 在当前位置创建一个 3x3 的小方框
    #             previous_pixel = output[c,row-1, col]
    #             local_pixel = output[c,row, col]
    #             if local_pixel != previous_pixel:
    #                 edges_img[c,row, col]=1
    return edges_img.transpose(1,0)
    #return 1-((2. * intersection + smooth) / \
    #    (torch.sum(output_) + torch.sum(target_) + smooth))

def edge_detection2(output):
    #vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
    # 定义横向过滤器
    #horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]
    # 得到图片的维数
    #output = output.transpose(0,1)
    n,m = output.shape
    # 初始化边缘图像
    edges_img = np.zeros(output.shape)#output.copy()
    # 循环遍历图片的全部像素
    for r in range(2,n):
        mask = (output[r,:]!=output[r-1,:])
        #print(mask)
        edges_img[r,mask]=1
    for p in range(2,m):
        mask = (output[:,p]!=output[:,p-1])
        #print(mask)
        edges_img[mask,p]=1
    # for c in range(channel):
    #     for row in range(2, n):
    #         for col in range(m):
    #             # 在当前位置创建一个 3x3 的小方框
    #             previous_pixel = output[c,row-1, col]
    #             local_pixel = output[c,row, col]
    #             if local_pixel != previous_pixel:
    #                 edges_img[c,row, col]=1
    return edges_img#.transpose(0,1)

def edge_detection3(output):
    #vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
    # 定义横向过滤器
    #horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]
    # 得到图片的维数
    #output = output.transpose(0,1)
    n,m = output.shape
    # 初始化边缘图像
    edges_img = np.zeros(output.shape)#output.copy()
    # 循环遍历图片的全部像素
    for r in range(2,n):
        mask = (output[r,:]!=output[r-1,:])
        #print(mask)
        edges_img[r,mask]=1
    for p in range(2,m):
        mask = (output[:,p]!=output[:,p-1])
        #print(mask)
        edges_img[mask,p]=1
    # for c in range(channel):
    #     for row in range(2, n):
    #         for col in range(m):
    #             # 在当前位置创建一个 3x3 的小方框
    #             previous_pixel = output[c,row-1, col]
    #             local_pixel = output[c,row, col]
    #             if local_pixel != previous_pixel:
    #                 edges_img[c,row, col]=1
    return edges_img#.transpose(0,1)

def per_class_assd(output, target,assds,num_classes):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # print(output.shape)
    # target = target /255
    # output = output* 255

    for k in range(1,num_classes):
        pred_test_data_tr = output.copy()
        pred_gt_data_tr = target.copy()
        # label_copy = np.zeros(pred_test_data_tr.shape, dtype=np.float32)
        # gt_copy = np.zeros(pred_gt_data_tr.shape, dtype=np.float32)
        pred_test_data_tr[pred_test_data_tr!=k] =0
        pred_gt_data_tr[pred_gt_data_tr!=k] =0
        if np.count_nonzero(pred_test_data_tr)!=0 and np.count_nonzero(pred_gt_data_tr)!=0:
            assd=mmb.asd(pred_test_data_tr, pred_gt_data_tr)
        else:
            assd = np.nan

        assds.append(assd)

    return assds

def precision_and_recall(y_true_in1, y_pred_in1):
    # y_true_in = y_true_in.astype('float32') * 255
    # y_pred_in = y_pred_in.astype('float32') * 255

    prec = []
    reca = []
    y_pred_in = y_pred_in1 > 0.5
    y_true_in = y_true_in1 > 0
    true_positives = np.sum((y_true_in * y_pred_in)==1)

    y_true_in1 = ((y_true_in+1)  * y_pred_in)
    false_positives = np.sum(y_true_in1 ==1)

    y_pred_in1 = ((y_pred_in+1)  * y_true_in)
    false_negatives = np.sum(y_pred_in1 ==1)

    y_pred_in2 = ((y_pred_in+1)  * (y_true_in+1))
    true_negatives = np.sum(y_pred_in2 ==1)

    if (true_positives + false_positives) > 0:
        p1 = true_positives / (true_positives + false_positives )
    else:
        p1 = 0
    if (true_positives + false_negatives) > 0:
        p2 = true_positives / (true_positives + false_negatives )
    else:
        p2 = 0

    return p1,p2

def update_variance(model, labels, pred1, pred2): #Recitfying Pseudo label
    # print(labels.shape)
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction = 'none')#loss_calc#
    kl_distance = nn.KLDivLoss( reduction = 'none')
    loss = criterion(pred1, labels) #ijcv Eq.3 L_ce
    sm = torch.nn.Softmax(dim = 1)
    log_sm = torch.nn.LogSoftmax(dim = 1)

    # print(pred1.shape,pred2.shape)
    #n, h, w = labels.shape
    #labels_onehot = torch.zeros(n, self.num_classes, h, w)
    #labels_onehot = labels_onehot.cuda()
    #labels_onehot.scatter_(1, labels.view(n,1,h,w), 1)
    tmp = kl_distance(log_sm(pred1),sm(pred2))
    # print('tmp.shape',tmp.shape)
    variance = torch.sum(tmp, dim=1)     #ijcv Eq.7 D_kl
    exp_variance = torch.exp(-variance)     #exp{-D_kl}
    #variance = torch.log( 1 + (torch.mean((pred1-pred2)**2, dim=1)))
    #torch.mean( kl_distance(self.log_sm(pred1),pred2), dim=1) + 1e-6
    # print(variance.shape,loss.shape,exp_variance.shape)    # torch.Size([10, 256, 256]) torch.Size([10, 256, 256]) torch.Size([10, 256, 256])
    print('variance min: %.4f'%torch.min(exp_variance[:]))
    print('variance max: %.4f'%torch.max(exp_variance[:]))
    print('variance sum: %.4f'%torch.sum(exp_variance[:]))
    print('variance mean: %.4f'%torch.mean(exp_variance[:]))
    #loss = torch.mean(loss/variance) + torch.mean(variance)
    loss = torch.mean(loss*exp_variance) + torch.mean(variance) #ijcv Eq.12 L_rect = exp{-D_kl}*L_ce+D_kl
    return loss,variance
