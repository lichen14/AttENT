# --------------------------------------------------------
# Domain adpatation evaluation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import os
import os.path as osp
import time
import seaborn as sns
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt # plt 用于显示图片
from utils.func import per_class_iu, fast_hist,per_class_iou_score,per_class_dice,per_class_assd
from utils.serialization import pickle_dump
from torchvision.utils import make_grid
import medpy.metric.binary as mmb
from skimage.io import imsave
import xlsxwriter as xw
USE_CUDA = torch.cuda.is_available()
def evaluate_domain_adaptation( models, test_loader, cfg,
                                fixed_test_size=True,
                                verbose=True):
    device = torch.device("cuda:0,1,2,3" if USE_CUDA else "cpu")
    interp = None
    models[0] = torch.nn.DataParallel(models[0], device_ids = [0,1,2,3])
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader, interp, fixed_test_size,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models,
                device, test_loader, interp,
                fixed_test_size, verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for index, batch in tqdm(enumerate(test_loader)):
        image, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_main = model(image.cuda(device))[1]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)


def eval_best(cfg, models,
              device, test_loader, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'

    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res.pkl')

    workbook = xw.Workbook(osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'result2.xlsx'))
    worksheet = workbook.add_worksheet('result2')
    all_res = {}
    cur_best_dice = -1
    cur_best_model = ''
    output_3d = np.zeros((len(test_loader),256,256))
    label_3d = np.zeros((len(test_loader),256,256))
    models_path = './experiments/snapshots/models_path.txt'
    with open(models_path) as f:
        model_ids = [i_id.strip() for i_id in f]


    for i_iter in range(len(model_ids)):
        # restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'best_model.pth')
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'best_model.pth')#model_ids[i_iter]#

        print("Evaluating model", restore_from)
        IoUs_all = np.zeros((cfg.NUM_CLASSES-1, len(test_loader)))
        DICEs_all = np.zeros((cfg.NUM_CLASSES-1, len(test_loader)))
        ASSDs_all = np.zeros((cfg.NUM_CLASSES-1, len(test_loader)))
        # if i_iter not in all_res.keys():
        load_checkpoint_for_evaluation(models[0], restore_from, device)
        # eval
        hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))

        test_iter = iter(test_loader)
        worksheet.write(i_iter, 0, model_ids[i_iter])
        os.makedirs(osp.join(cfg.TEST.SNAPSHOT_DIR[0]+'/output1/'), exist_ok=True)
        os.makedirs(osp.join(cfg.TEST.SNAPSHOT_DIR[0]+'/entropy1/'), exist_ok=True)
        assd_list = []
        dice_list = []
        for index,(image, label, _,name) in tqdm(enumerate(test_loader), total=len(test_loader)):

            if not fixed_test_size:
                interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
            with torch.no_grad():
                pred_main = models[0](image.cuda(device))[1]
                output1 = interp(pred_main).cpu().data[0].numpy()
                output2 = output1.transpose(1, 2, 0)
                output = np.argmax(output2, axis=2) #(256, 256)

                output_sm = F.softmax(interp(pred_main)).cpu().data[0].numpy().transpose(1, 2, 0)
                output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                                    keepdims=False)  /np.log2(cfg.NUM_CLASSES)                                        # self-information maps I_x
                grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                                       range=(0, np.log2(cfg.NUM_CLASSES)))
                # print(output_sm.shape,output_ent.shape,grid_image.shape) #(256, 256, 5) (256, 256) torch.Size([3, 256, 256])


            label = label.numpy()[0]
            output_3d[index]= output
            label_3d[index] = label
            for c in range(1, cfg.NUM_CLASSES):
                pred_test_data_tr = output.copy()
                pred_test_data_tr[pred_test_data_tr != c] = 0

                pred_gt_data_tr = label.copy()
                pred_gt_data_tr[pred_gt_data_tr != c] = 0

                if np.count_nonzero(pred_test_data_tr)!=0 and np.count_nonzero(pred_gt_data_tr)!=0:
                    assd_list.append(mmb.asd(pred_test_data_tr, pred_gt_data_tr))

                else:
                    assd_list.append(np.nan)

            iou_single=[]
            iou_single = per_class_iou_score(output, label,iou_single,cfg.NUM_CLASSES)
            IoUs_all[:,index]=iou_single
            dice_single=[]
            dice_single = per_class_dice(output, label,dice_single,cfg.NUM_CLASSES)
            DICEs_all[:,index]=dice_single
            assd_single=[]
            assd_single = per_class_assd(output, label,assd_single,cfg.NUM_CLASSES)
            ASSDs_all[:,index]=assd_single

            hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)

            if verbose :#and index > 0:# and index % 100 == 0:
                print('name is:',name,'dice : {:0.2f},assd: {:0.2f}'.format(
                    round(np.nanmean(dice_single) * 100, 2), round(np.nanmean(assd_single), 2)))

            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)

        my_computed_dice = round(np.nanmean(DICEs_all) * 100, 2)
        my_computed_assd = round(np.nanmean(ASSDs_all), 2)

        my_computed_3D_dice = per_class_dice(output_3d, label_3d,[],cfg.NUM_CLASSES)
        my_computed_3D_assd = per_class_assd(output_3d, label_3d,[],cfg.NUM_CLASSES)


        if cur_best_dice < my_computed_dice:
            cur_best_dice = my_computed_dice
            cur_best_assd = my_computed_assd
            cur_best_dice_all = DICEs_all
            cur_best_assd_all = ASSDs_all
            dice_best_liver = round(np.nanmean(DICEs_all[3,:]) * 100, 2)
            dice_best_rightK = round(np.nanmean(DICEs_all[2,:]) * 100, 2)
            dice_best_leftK = round(np.nanmean(DICEs_all[1,:]) * 100, 2)
            dice_best_spleen = round(np.nanmean(DICEs_all[0,:]) * 100, 2)

            assd_best_liver = round(np.nanmean(ASSDs_all[3,:]), 2)
            assd_best_rightK = round(np.nanmean(ASSDs_all[2,:]), 2)
            assd_best_leftK = round(np.nanmean(ASSDs_all[1,:]), 2)
            assd_best_spleen = round(np.nanmean(ASSDs_all[0,:]), 2)

        # print('\tCurrent best Dice-average:', cur_best_dice)
        # print('\tCurrent best Dice-liver:', dice_best_liver)
        # print('\tCurrent best Dice-rightKidney:', dice_best_rightK)
        # print('\tCurrent best Dice-leftKidney:', dice_best_leftK)
        # print('\tCurrent best Dice-spleen:', dice_best_spleen)
        # print('\tCurrent best assd-average:', cur_best_assd)
        # print('\tCurrent best assd-liver:', assd_best_liver)
        # print('\tCurrent best assd-rightKidney:', assd_best_rightK)
        # print('\tCurrent best assd-leftKidney:', assd_best_leftK)
        # print('\tCurrent best assd-spleen:', assd_best_spleen)
        # print('\tCurrent best my_computed_3D_assd:', my_computed_3D_assd)
        # print('\tCurrent best my_computed_3D_AVG_assd:', np.mean(my_computed_3D_assd))

        worksheet.write(i_iter, 1, np.nanmean(my_computed_3D_dice)*100)
        worksheet.write(i_iter, 2, np.nanmean(my_computed_3D_assd))
    workbook.close()
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, cur_best_dice_all,cur_best_assd_all)

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, DICEs_all,ASSDs_all):
    # print(DICEs_all.shape,ASSDs_all.shape)
    for ind_class in range(cfg.NUM_CLASSES-1):
        print(name_classes[1+ind_class]
              + '\t' + str(round(np.nanmean(DICEs_all[cfg.NUM_CLASSES-2-ind_class,:]) * 100, 2))
              + '\t' + str(round(np.nanmean(ASSDs_all[cfg.NUM_CLASSES-2-ind_class,:]), 2)))
