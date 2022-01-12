import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt # plt 用于显示图片

def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()

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

    target_mask = (target >= 0) * (target != 255)   #??
    target = target[target_mask]                    #??
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()  #Predict(n,h,w,c)
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)  #将target视为n,h,w,c，重复填充最后的c通道


    loss = F.cross_entropy(predict, target, size_average=True)
    # loss = F.cross_entropy(tt, target, size_average=True)

    return loss


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))
