# --------------------------------------------------------
# Domain adpatation training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import os
import sys
from pathlib import Path
from PIL import Image
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt # plt 用于显示图片
from model.discriminator import get_fc_discriminator
from utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from utils.func import loss_calc, bce_loss
from utils.func import prob_2_entropy
from utils.viz_segmask import colorize_mask
from utils.func import per_class_iou_score,per_class_dice,per_class_assd
import medpy.metric.binary as mmb
from torch.optim import lr_scheduler

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0,1,2,3" if USE_CUDA else "cpu")
def train_attent(model, d_aux,d_main,trainloader, targetloader, cfg,epoch):
    ''' UDA training with attent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    # device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # OPTIMIZERS
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters(cfg.TRAIN.LEARNING_RATE)),
    #                       lr=cfg.TRAIN.LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(cfg.TRAIN.LEARNING_RATE), lr=cfg.TRAIN.LEARNING_RATE,
                                 betas=(0.9, 0.99))
    # optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
    #                       lr=cfg.TRAIN.LEARNING_RATE,
    #                       momentum=cfg.TRAIN.MOMENTUM,
    #                       weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))



    # adapt LR if needed


    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)
    sm = torch.nn.Softmax(dim = 1)
    log_sm = torch.nn.LogSoftmax(dim = 1)
    kl_distance = nn.KLDivLoss( reduction = 'none')
    # labels for adversarial training
    source_LABEL = 1
    target_LABEL = 0
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    # print(len(targetloader))
    for i_iter,(images, target_labels, _, target_names) in tqdm(enumerate(targetloader), total=len(targetloader)):#range(cfg.TRAIN.EARLY_STOP + 1)):
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # reset optimizers
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)
        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False

        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch

        pred_src_aux, pred_src_main = model(images_source.to(device)) #输出两层的分割预测结果

        # print(images_source.shape, labels.shape)

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)

        loss_seg_src_main = loss_calc(pred_src_main, labels, device) #计算aux和main两种分割预测结果的loss
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)  #交叉熵合并后误差反向传播
        loss.backward(retain_graph=True) #Lseg(x_s,y_s) 源域分割损失
        # scheduler.step(loss.item())
        # adversarial training to fool the discriminator
         #images: torch.Size([1, 3, 256, 256]) target_labels torch.Size([1, 256, 256])

        # print('images:',images.shape,'target_labels',target_labels.shape)
        pred_trg_aux, pred_trg_main = model(images.to(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            # print('pred_trg_aux',pred_trg_aux.shape)    #[1, 2, 256, 256]
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))#输出鉴别器结果
            # print('d_out_aux---aux',np.mean(d_out_aux.cpu().data.numpy()))  #[1, 1, 8, 8]
            loss_adv_trg_aux = bce_loss(d_out_aux, source_LABEL)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main)    #[1, 2, 256, 256]
        pseudo_aux = pred_trg_aux#.copy()
        pseudo_main = pred_trg_main#.copy()
        pseudo_label = np.argmax((pseudo_main+0.5*pseudo_aux).data.cpu(),axis=1)
        # print(pseudo_label.shape)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))   #softmax得到0-1的概率prob，prob_2_entropy转化为self-information I_x
        # print('d_out_main',d_out_main)    #[1, 1, 8, 8]

        loss_adv_trg_main = bce_loss(d_out_main, source_LABEL)
        # print('d_out_main---mean：',np.mean(d_out_main.cpu().data.numpy()))    #[1, 1, 8, 8]

        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        loss = loss
        loss.backward(retain_graph=True) #以上是公式(8)的损失
        
        # Train discriminator networks训练鉴别器网络
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True

        # train with source 在源域的损失
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()    #不会对新的pred_scr_aux求梯度
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
            loss_d_aux = bce_loss(d_out_aux, source_LABEL)
            loss_d_aux = loss_d_aux / 2 # 2是长度，=|X_s|
            loss_d_aux.backward()
        #
        else:
            loss_d_aux = 0
        #
        pred_src_main = pred_src_main.detach()    #不会对新的pred_src_main求梯度
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_LABEL)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target 在目标域的损失
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()    #不会对新的pred_tar_aux求梯度
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_LABEL)
            loss_d_aux = loss_d_aux / 2 # 2是长度，=|X_t|
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()  #不会对新的pred_tar_main求梯度
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_LABEL)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()
        #
        optimizer.step()    #分割网络的优化器
        #分割网络的优化器
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()
        #
        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,   #L_seg(x_s,y_s) 公式(9)前部分
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,   #L_D(I_x_t,1) 公式(9)后部分
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}   #L_D(I_x_s,1)+L_D(I_x_t,0) 公式(7)
        
        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, epoch*len(trainloader)+i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE ==  1:#cfg.TRAIN.TENSORBOARD_VIZRATE - 1:

                draw_in_tensorboard(writer, images[0], epoch*len(trainloader)+i_iter, pred_trg_main, target_labels,num_classes, 'T') #[3, 256, 256]),[1, 5, 256, 256],[1, 256, 256, 3]
                draw_in_tensorboard(writer, images_source[0], epoch*len(trainloader)+i_iter, pred_src_main, labels,num_classes, 'S')  #[1, 3, 256, 256],[1, 2, 256, 256]
    return model

def validate_attent(model, train_loader, validloader, cfg,d_main,d_aux,epoch,cur_best_dice,cur_best_model):
    ''' UDA validate with attent
    '''
    # Create the model and start the training.
    cur_best_model_state = model
    print(epoch,' cur_best_dice:',cur_best_dice)
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    # device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    #begin validate
    model.eval()

    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    IoUs_all = np.zeros((cfg.NUM_CLASSES-1, len(validloader)))
    DICEs_all = np.zeros((cfg.NUM_CLASSES-1, len(validloader)))
    ASSDs_all = np.zeros((cfg.NUM_CLASSES-1, len(validloader)))
    for index,(images, target_labels, _, _) in tqdm(enumerate(validloader), total=len(validloader)):

        # print('batch is :',batch
        interp = nn.Upsample(size=(target_labels.shape[1], target_labels.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            pred_main = model(images.cuda(device))[1]
            output = interp(pred_main).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)

        label = target_labels.numpy()[0]

        iou_single=[]
        iou_single = per_class_iou_score(output, label,iou_single,cfg.NUM_CLASSES)
        # print(iou_single)
        IoUs_all[:,index]=iou_single
        # print(IoUs_all)
        dice_single=[]
        dice_single = per_class_dice(output, label,dice_single,cfg.NUM_CLASSES)
        DICEs_all[:,index]=dice_single

        assd_single=[]
        assd_single = per_class_assd(output, label,assd_single,cfg.NUM_CLASSES)
        ASSDs_all[:,index]=assd_single

    restore_from = osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{epoch}.pth')
    my_computed_miou = round(np.nanmean(IoUs_all) * 100, 2)
    my_computed_dice = round(np.nanmean(DICEs_all) * 100, 2)
    my_computed_assd = round(np.nanmean(ASSDs_all), 2)
    # torch.save(model.state_dict(), restore_from)
    if cur_best_dice < my_computed_dice:
        cur_best_dice = my_computed_dice
        cur_best_assd = my_computed_assd
        cur_best_model = restore_from
        cur_best_dice_all = DICEs_all
        cur_best_assd_all = ASSDs_all

        print('taking snapshot ...')
        print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
        snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
        torch.save(model.state_dict(), snapshot_dir / f'best_model_{epoch}.pth')
        cur_best_model_state = model


    # print('\tCurrent mIoU:', computed_miou)
    # print('\tCurrent my mIoU:', my_computed_miou)
    # print('\tCurrent my dice:', my_computed_dice)
    # print('\tCurrent my assd:', my_computed_assd)
    # print('\tCurrent best model:', cur_best_model)
    # print('\tCurrent best dice:', cur_best_dice)

    return cur_best_model_state, cur_best_model,cur_best_dice
        
def draw_in_tensorboard(writer, images, i_iter, pred_main, target_labels,num_classes, type_):
    grid_image = make_grid(images.clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(
            colorize_mask(np.asarray(
                                np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),axis=2),
                                dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    # print('target_labels:',target_labels.shape)#[1, 256, 256]
    grid_image = make_grid(target_labels[:1].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'GroundTrue - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)                                          # self-information maps I_x
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def train_supervised(model, trainloader, targetloader, cfg,epoch):
    '''
    training without UDA
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    # device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    # model = torch.nn.DataParallel(model, device_ids = [0,1,2]).to(device)
    # print('Model DataParalleled')


    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters(cfg.TRAIN.LEARNING_RATE)),
                          lr=cfg.TRAIN.LEARNING_RATE)
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    # optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
    #                              betas=(0.9, 0.99))
    # optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
    #                               betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_LABEL = 1
    target_LABEL = 0
    trainloader_iter = enumerate(trainloader)
    # targetloader_iter = enumerate(targetloader)
    for i_iter,(images_source, labels, _, _) in tqdm(enumerate(trainloader), total=len(trainloader)):#range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        
        pred_src_aux, pred_src_main = model(images_source.cuda(device)) #输出两层的分割预测结果

        # print(images_source.shape, labels.shape)

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        
        loss_seg_src_main = loss_calc(pred_src_main, labels, device) #计算aux和main两种分割预测结果的loss
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)  #交叉熵合并后误差反向传播
        loss.backward() #Lseg(x_s,y_s) 源域分割损失

        optimizer.step()    
        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main}   #L_D(I_x_s,1)+L_D(I_x_t,0) 公式(7)
        
        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, epoch*len(trainloader)+i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE ==  1:#cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                # print('image-tenosrboardx**************************************************')
                # print(images[0].shape,pred_trg_main.shape)    #[3, 256, 256])[1, 5, 256, 256])
                print_losses(current_losses, i_iter)
                # draw_in_tensorboard(writer, images[0], epoch*len(trainloader)+i_iter, pred_trg_main, target_labels,num_classes, 'T') #[3, 256, 256]),[1, 5, 256, 256],[1, 256, 256, 3]
                draw_in_tensorboard(writer, images_source[0], epoch*len(trainloader)+i_iter, pred_src_main, labels,num_classes, 'S')  #[1, 3, 256, 256],[1, 2, 256, 256]
    return model

def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_domain_adaptation(model, trainloader, targetloader,validloader, cfg):

    model = torch.nn.DataParallel(model, device_ids = [0,1,2,3])
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes= cfg.NUM_CLASSES)
    d_aux.train()
    d_aux.to(device)

    # seg maps, 即 output, level
    d_main = get_fc_discriminator(num_classes=cfg.NUM_CLASSES)
    d_main.train()
    d_main.to(device)
    cur_best_dice=-1
    cur_best_model_path =''
    cur_best_model = model
    for epoch in range(cfg.TRAIN.EPOCHS):
        print('Epoch [%d/%d]' %(epoch, cfg.TRAIN.EPOCHS))
        if cfg.TRAIN.DA_METHOD == 'attent':
            # print('begin train')
            epoch_best_model = train_attent(cur_best_model,d_aux,d_main, trainloader, targetloader, cfg,epoch)
            epoch_best_model,epoch_best_model_path,current_best_dice = validate_attent(epoch_best_model, targetloader, validloader, cfg,d_main,d_aux,epoch,cur_best_dice,cur_best_model_path)
        elif cfg.TRAIN.DA_METHOD == 'supervised':
            epoch_best_model = train_supervised(cur_best_model,trainloader,targetloader,cfg,epoch)
            epoch_best_model,epoch_best_model_path,current_best_dice = validate_attent(epoch_best_model, trainloader, validloader, cfg,d_main,d_aux,epoch,cur_best_dice,cur_best_model_path)
        else:
            raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
        cur_best_dice = current_best_dice
        cur_best_model_path = epoch_best_model_path
        cur_best_model = epoch_best_model
