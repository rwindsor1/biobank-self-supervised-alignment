''' Train 2d MRI spine segmenter with 2d DXA labels
Rhydian Windsor 07/02/20
'''

import sys, os, glob
sys.path.append('/users/rhydian/self-supervised-project')
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import auc

from models.UNet3D import UNet3D
from datasets.BothMidCoronalDataset import BothMidCoronalDataset, BothMidCoronalHardNegativesDataset
from gen_utils import balanced_l1w_loss, grayscale, red,blue,yellow, dice_and_iou, green
from loss_functions import SSECLoss
from sacred import Experiment
from sacred.observers import MongoObserver
from models.UNet2D import UNet
# to make sensible stop
from sacred import SETTINGS
from torch.utils.data.dataloader import default_collate
ex = Experiment('DXAtoMRISpineSegmenter')
ex.captured_out_filter = lambda captured_output: "output capturing turned off."
ex.observers.append(MongoObserver(url='login1.triton.cluster:27017', db_name='SelfSupervisedProject'))

@ex.config
def expt_config():
    ADAM_BETAS = (0.9,0.999)
    NUM_WORKERS = 20
    TRAIN_NUM_WORKERS = NUM_WORKERS
    VAL_NUM_WORKERS   = NUM_WORKERS
    TEST_NUM_WORKERS  = NUM_WORKERS
    # Scan Encoder Details
    BATCH_SIZE=6
    TRAIN_BATCH_SIZE = BATCH_SIZE
    VAL_BATCH_SIZE = BATCH_SIZE
    TEST_BATCH_SIZE = BATCH_SIZE
    USE_CUDA=True
    VALIDATION_ITERATIONS = 100
    LR=0.0001
    WEIGHTS_PATH='../model_weights/spine_segmenter_2d'
    USE_TEMP=True
    if USE_TEMP:
        DATASET_ROOT = '/tmp/rhydian'
    else:
        DATASET_ROOT = '/scratch/shared/beegfs/rhydian/UKBiobank'

    SEG_PARTS = ['spine','pelvis','pelvic_cavity']



def my_collate_fn(batch):
    ret_dict = default_collate(batch)
    mri_targets = []
    for i in range(len(batch)):
        mri_target = TF.affine(TF.rotate(ret_dict['target'][i][None], ret_dict['angle'][i].item(), center=(0,0)),0,[ret_dict['t_y'][i].item(),ret_dict['t_x'][i].item()],1,(0,0))[:,:,:ret_dict['mri_img'].shape[-2], :ret_dict['mri_img'].shape[-1]]
        mri_targets.append(mri_target)
    ret_dict['mri_target'] = torch.cat(mri_targets,dim=0)
    ret_dict['mri_target'] = ret_dict['mri_target'][:,[1,2,-1]]
    return ret_dict

@ex.capture
def get_dataloaders(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE, 
                    TRAIN_NUM_WORKERS, VAL_NUM_WORKERS, TEST_NUM_WORKERS, 
                    DATASET_ROOT):
    train_ds = BothMidCoronalDataset(set_type='train', all_root=DATASET_ROOT, single_dxa=False, single_mri=False, return_segmentations=True,equal_res=True, augment=False, relative_scan_alignments='/users/rhydian/self-supervised-project/scan-registration/scan_relative_transforms.csv')
    val_ds   = BothMidCoronalDataset(set_type='val'  , all_root=DATASET_ROOT, single_dxa=False, single_mri=False, return_segmentations=True,equal_res=True, augment=False, relative_scan_alignments='/users/rhydian/self-supervised-project/scan-registration/scan_relative_transforms.csv')
    test_ds  = BothMidCoronalDataset(set_type='test' , all_root=DATASET_ROOT, single_dxa=False, single_mri=False, return_segmentations=True,equal_res=True, augment=False, relative_scan_alignments='/users/rhydian/self-supervised-project/scan-registration/scan_relative_transforms.csv')

    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, collate_fn=my_collate_fn, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=VAL_BATCH_SIZE,   num_workers=VAL_NUM_WORKERS,   collate_fn=my_collate_fn, shuffle=False, drop_last=True)
    test_dl  = DataLoader(test_ds,  batch_size=TEST_BATCH_SIZE,  num_workers=TEST_NUM_WORKERS,  collate_fn=my_collate_fn, shuffle=False, drop_last=True)
    return train_dl, val_dl, test_dl




@ex.capture
def load_model(WEIGHTS_PATH, USE_CUDA,SEG_PARTS):
    model = UNet(2,len(SEG_PARTS))
    model = nn.DataParallel(model)
    model_dict = model.state_dict()
    print(f'Trying to load from {WEIGHTS_PATH}')
    if not os.path.isdir(WEIGHTS_PATH):
        os.mkdir(WEIGHTS_PATH)
    # load model weights
    if os.path.isdir(WEIGHTS_PATH):
            list_of_pt = glob.glob(WEIGHTS_PATH + '/*.pt')
            if len(list_of_pt):
                latest_pt = max(list_of_pt, key=os.path.getctime)
                checkpoint = torch.load(latest_pt, map_location=torch.device('cpu'))
                model_dict.update(checkpoint['model_weights'])
                model.load_state_dict(model_dict)
                best_loss = checkpoint['best_loss']
                epochs = checkpoint['epochs'] 
                print(f"==> Resuming model trained for {epochs} epochs...")
            else:
                print("==> Training Fresh Model ")
                best_loss = 99999999999999
                epochs = 0
    else:
        raise Exception(f'Could not find directory at {WEIGHTS_PATH}')
    if USE_CUDA:
        model.cuda()

    return model, best_loss, epochs


@ex.capture
def save_models(model, best_loss, epochs, WEIGHTS_PATH):
    print(f'==> Saving Model Weights to {WEIGHTS_PATH}')
    state = {'model_weights': model.state_dict(),
             'best_loss'        : best_loss,
             'epochs'           : epochs
            }
    if not os.path.isdir(WEIGHTS_PATH):
        os.mkdir(WEIGHTS_PATH)
    previous_checkpoints = glob.glob(WEIGHTS_PATH + '/ckpt*.pt', recursive=True)
    for previous_checkpoint in previous_checkpoints:
       os.remove(previous_checkpoint)
    torch.save(state, WEIGHTS_PATH + '/ckpt' + str(epochs) + '.pt')
    return

@ex.capture
def get_optim(model, LR, ADAM_BETAS):
    optim = Adam(model.parameters(), lr=LR, betas=ADAM_BETAS)
    return optim

def dice_loss(out, target):
    out = torch.sigmoid(out)
    numerators = (out * target).flatten(start_dim=1).sum(dim=1)
    ps = (out).flatten(start_dim=1).sum(dim=1)
    ns = (target).flatten(start_dim=1).sum(dim=1)
    dice_coeffs = 2*numerators/(ps+ns+1)
    dice_coeff = dice_coeffs.mean()
    return 1-dice_coeff

    
@ex.capture
def epoch(dl, model, optim, SEG_PARTS, USE_CUDA, validate=False):
    if validate: model.eval()
    else: model.train()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = dice_loss
    losses = []
    dices = {}
    ious = {}
    for idx, part in enumerate(SEG_PARTS): dices[part] = []; ious[part]  = []
    dices['avg'] = []; ious['avg']=[]

    pbar = tqdm(dl)
    for idx, sample in enumerate(pbar):
        optim.zero_grad()
        if USE_CUDA:
            sample['mri_img'] = sample['mri_img'].cuda()
            sample['mri_target'] = sample['mri_target'].cuda()

        out = model(sample['mri_img'])
        target = sample['mri_target']

        l_spine = criterion(out[:,0], target[:,0])
        l_pelvi = criterion(out[:,1], target[:,1])
        l_pel_c = criterion(out[:,2], target[:,2])
        loss = (l_spine+l_pelvi+l_pel_c)/3
        if not validate:
            loss.backward()
            optim.step()
        losses.append(loss.item())

        batch_avg_dices = []
        batch_avg_ious = []
        for idx, part in enumerate(SEG_PARTS):
            dice, iou = dice_and_iou(out[:,idx], target[:,idx])
            dices[part].append(dice)
            ious[part].append(iou)
            batch_avg_dices.append(dice)
            batch_avg_ious.append(iou)

        dices['avg'].append(np.mean(batch_avg_dices));ious['avg'].append(batch_avg_ious)
        pbar.set_description(f"Loss: {loss.item():.4} Dice: {dices['avg'][-1]:.4}")
    for key in dices:
        dices[key] = np.mean(dices[key])
    return np.mean(losses), dices


@ex.command(unobserved=True)
def show_predictions():
    train_dl, val_dl, test_dl = get_dataloaders()
    model, best_loss, epochs = load_model()
    model.eval()

    pbar = tqdm(test_dl)
    for idx, sample in enumerate(pbar):
        with torch.no_grad():
            out = torch.sigmoid(model(sample['mri_img']))
        for batch_idx in range(sample['mri_img'].shape[0]):
            plt.figure(figsize=(10,4))
            plt.subplot(131)
            plt.axis('off')
            plt.imshow(grayscale(sample['dxa_img'][batch_idx,0])/2+red(sample['target'][batch_idx,1])+blue(sample['target'][batch_idx,2])-red(sample['target'][batch_idx,2])-green(sample['target'][batch_idx,2])-blue(sample['target'][batch_idx,-1])+2*green(sample['target'][batch_idx,-1]))
            plt.subplot(132)
            plt.axis('off')
            plt.title('Transferred Segmentation Map')
            plt.imshow(grayscale(sample['mri_img'][batch_idx,0])+red(sample['mri_target'][batch_idx,0])+blue(sample['mri_target'][batch_idx,1])-blue(sample['mri_target'][batch_idx,2])+green(sample['mri_target'][batch_idx,2]))
            plt.subplot(133)
            plt.axis('off')
            plt.title('Prediction')
            plt.imshow(grayscale(sample['mri_img'][batch_idx,0])+red(out[batch_idx,0])+blue(out[batch_idx,1])-blue(out[batch_idx,2])+green(out[batch_idx,2]))
            plt.savefig(f'example-segmentations/test_{idx}_{batch_idx}.png')
            plt.close('all')

            plt.imshow(grayscale(sample['mri_img'][batch_idx,0]))
            plt.savefig(f'bare_{idx}_{batch_idx}')
            os.system(f'imgcat example-segmentations/test_{batch_idx}.png')

@ex.automain
def main(_run):
    train_dl, val_dl, test_dl = get_dataloaders()
    model, best_loss, epochs = load_model()
    optim = get_optim(model)
    while True:
        print('Training...')
        train_loss, dices = epoch(train_dl, model, optim, validate=False)
        for key in dices:
            print(key, dices[key])
            _run.log_scalar(f'training.{key}', dices[key])
        val_loss, dices  = epoch(val_dl,model, optim, validate=True)
        print('Validating...')
        for key in dices:
            print(key, dices[key])
            _run.log_scalar(f'validation.{key}', dices[key])

        if val_loss < best_loss:
            save_models(model, val_loss, epochs)
            best_loss = val_loss
        epochs += 1
