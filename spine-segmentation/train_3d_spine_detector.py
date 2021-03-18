''' Train 3d spine segmenter with 2d labels
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
from dataset import SpineSegmentTrainDataset, SpineSegmentTestDataset
from gen_utils import balanced_l1w_loss, grayscale, red
from loss_functions import SSECLoss
from sacred import Experiment
from sacred.observers import MongoObserver
# to make sensible stop
from sacred import SETTINGS

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
    NOTE=''
    LR=0.001
    WEIGHTS_PATH='../model_weights/spine_segmenter_flipped'
    USE_TEMP=True
    USE_FLIP=True
    USE_HALF_PRECISION=False
    if USE_TEMP:
        DATASET_ROOT = '/tmp/rhydian'
    else:
        DATASET_ROOT = '/scratch/shared/beegfs/rhydian/UKBiobank'



@ex.capture
def get_dataloaders(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE,
                    TRAIN_NUM_WORKERS, VAL_NUM_WORKERS, TEST_NUM_WORKERS,
                    DATASET_ROOT):

    train_ds = SpineSegmentTrainDataset(set_type='train', all_root=DATASET_ROOT)
    val_ds   = SpineSegmentTestDataset(all_root=DATASET_ROOT, set_type='val')
    test_ds  = SpineSegmentTestDataset(all_root=DATASET_ROOT, set_type='test')

    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,  batch_size=VAL_BATCH_SIZE,  num_workers=VAL_NUM_WORKERS,  shuffle=False, drop_last=True)
    test_dl  = DataLoader(test_ds,  batch_size=TEST_BATCH_SIZE,  num_workers=TEST_NUM_WORKERS,  shuffle=False, drop_last=True)
    return train_dl, val_dl, test_dl

@ex.capture
def get_optim(model, LR, ADAM_BETAS):
    optim = Adam(model.parameters(), lr=LR, betas=ADAM_BETAS)
    return optim

@ex.capture
def load_model(WEIGHTS_PATH, USE_CUDA, USE_HALF_PRECISION):
    model = UNet3D(1,1)
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
    if USE_HALF_PRECISION:
        model.half()  # convert to half precision
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    return model, best_loss, epochs

def get_dice(x,y):
    tp = ((x > 0)*((y==1))).sum()
    fp = ((x>0)*(y!=1)).sum()
    fn = ((x<0)*(y==1)).sum()
    return tp/(2*tp+fp+fn+1e-9)

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def balanced_bce_loss_with_logits(input, target, balanced=True):
    num_pos = (target == 1).sum()
    num_neg = (target == 0).sum()
    p_weight = (num_neg+1)/(num_pos+num_neg)
    n_weight = (num_pos+1)/(num_pos+num_neg)
    if balanced:
        weights = torch.zeros_like(target).float()
        weights[target == True] = p_weight
        weights[target == False] = n_weight
    else:
        weights = torch.ones_like(target).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(input, target, weights) 
    return loss

@ex.capture
def train_epoch(dl, model,optim,USE_CUDA,USE_FLIP):
    model.train()
    pbar = tqdm(dl)
    losses = []
    dices_2d = []
    criterion = balanced_bce_loss_with_logits
    train_stats = {}
    for idx, sample in enumerate(pbar):
        import pdb; pdb.set_trace()
        if USE_FLIP: flip = (np.random.random()>0.5)
        else: flip =False
        
        optim.zero_grad()
        if USE_CUDA:
            for key in sample:
                sample[key] = sample[key].cuda()
        target = sample['target_2d']
        if flip:
            vol = sample['vol'].permute(0,1,2,4,3)
            out = model(vol).permute(0,1,2,4,3)
        else:
            out = model(sample['vol'])
        projected_logits = F.max_pool3d(out,(1,out.shape[-2],1)).squeeze(-2)
        loss = criterion(projected_logits, target, balanced=False)
        # loss = dice_loss(projected_logits, target)
        loss.backward()
        optim.step()
        dice = get_dice(projected_logits, target)
        losses.append(loss.item())
        dices_2d.append(dice.item())
        pbar.set_description(f"loss: {loss:.3}, dice: {dice:.3}")
        if idx > 200:
            break
    train_stats['loss'] = np.mean(losses)
    train_stats['dices_2d'] = np.mean(dices_2d)
    return train_stats 

@ex.capture
def val_epoch(dl, model, USE_CUDA, show_results=False):
    model.eval()
    pbar = tqdm(dl)
    losses = []
    dices_3d = []
    dices_2d = []
    criterion = balanced_bce_loss_with_logits
    val_stats = {}
    for idx, sample in enumerate(pbar):
        if USE_CUDA:
            for key in sample:
                sample[key] = sample[key].cuda()
        with torch.no_grad():
            out = model(sample['vol'])
        import pdb; pdb.set_trace()
        target = sample['target_3d']
        projected_target = (target.sum(dim=-2)>=1).float()
        #projected_logits = F.max_pool3d(out,(1,out.shape[-2],1)).squeeze(-2)
        projected_logits = F.max_pool3d(out,(1,out.shape[-2],1)).squeeze(-2)
        loss = criterion(projected_logits, projected_target)
        # loss = dice_loss(projected_logits, target)
        dice_3d = get_dice(out, target)
        dice_2d = get_dice(projected_logits, projected_target)
        losses.append(loss.item())
        dices_3d.append(dice_3d.item())
        dices_2d.append(dice_2d.item())
        pbar.set_description(f"loss: {loss:.3}, 3D dice: {dice_3d.item():.3}, 2D dice: {dice_2d.item():.3}")
        if show_results:
            for batch_idx in range(sample['vol'].shape[0]):
                plt.figure(figsize=(10,10))
                mid_cor = sample['target_3d'][batch_idx].sum(dim=-1).sum(dim=-2)[0].argmax()
                mid_sag = sample['target_3d'][batch_idx].sum(dim=-2).sum(dim=-2)[0].argmax()
                plt.subplot(131)
                plt.title('image')
                plt.imshow(grayscale(sample['vol'][batch_idx,0,:,mid_cor,:]))
                plt.subplot(132)
                plt.title('coronal')
                plt.imshow(grayscale(sample['vol'][batch_idx,0,:,mid_cor,:])+ red(torch.sigmoid(out[batch_idx].max(dim=-2)[0][0])))
                plt.subplot(133)
                plt.title('sagittal')
                plt.imshow(grayscale(sample['vol'][batch_idx,0,:,:,mid_sag])+ red(torch.sigmoid(out[batch_idx].max(dim=-1)[0][0])))
                plt.savefig('test.png')
                plt.close('all')
                os.system('imgcat test.png')
    val_stats['loss'] = np.mean(losses)
    val_stats['dices_3d'] = np.mean(dices_3d)
    val_stats['dices_2d'] = np.mean(dices_2d)
    return val_stats



@ex.command(unobserved=True)
def view_training_examples():
    train_dl, val_dl, test_dl = get_dataloaders()
    model, best_loss, epochs = load_model()
    model.eval()
    for idx, sample in enumerate(train_dl):
        import pdb; pdb.set_trace()
        with torch.no_grad():
            out = model(sample['vol'])

        mid_sag = int(sample['vol'].shape[-1]/2)
        for batch_idx in range(sample['vol'].shape[0]):
            mid_cor = int(out[batch_idx,0].sum(dim=-1).sum(dim=-2).argmax())
            plt.figure(figsize=(4,10))
            plt.subplot(221)
            plt.title('source')
            plt.imshow(grayscale(sample['vol'][batch_idx,0,:,mid_cor,:]))
            plt.subplot(222)
            plt.title('predictions - sagittal')
            plt.imshow(grayscale(sample['vol'][batch_idx,0,:,:,mid_sag])+ red(torch.sigmoid(out[batch_idx,:,:,:,mid_sag][0])))
            plt.subplot(223)
            plt.imshow(grayscale(sample['vol'][batch_idx,0,:,mid_cor,:])+ red(torch.sigmoid(out[batch_idx,:,:,mid_cor,:][0])))
            plt.title('predictions - coronal')
            plt.subplot(224)
            plt.title('target')
            plt.imshow(grayscale(sample['vol'][batch_idx,0,:,mid_cor,:])+ red(sample['target_2d'][batch_idx,0]))
            plt.savefig('test.png')
            plt.close('all')
            os.system('imgcat test.png')
        continue

@ex.command(unobserved=True)
def view_results():
    train_dl, val_dl, test_dl = get_dataloaders()
    model, best_loss, epochs = load_model()
    val_stats = val_epoch(val_dl, model, show_results=True)
    return

@ex.command(unobserved=True)
def test():
    train_dl, val_dl, test_dl = get_dataloaders()
    model, best_loss, epochs = load_model()
    val_stats = val_epoch(test_dl, model)
    print(val_stats)
    return

@ex.capture
def save_models(model, val_stats, epochs, WEIGHTS_PATH):
    print(f'==> Saving Model Weights to {WEIGHTS_PATH}')
    state = {'model_weights': model.state_dict(),
             'best_loss'        : val_stats['loss'],
             'epochs'           : epochs
            }
    if not os.path.isdir(WEIGHTS_PATH):
        os.mkdir(WEIGHTS_PATH)
    previous_checkpoints = glob.glob(WEIGHTS_PATH + '/ckpt*.pt', recursive=True)
    for previous_checkpoint in previous_checkpoints:
       os.remove(previous_checkpoint)
    torch.save(state, WEIGHTS_PATH + '/ckpt' + str(epochs) + '.pt')
    return



@ex.automain
def main(USE_TEMP, _run):
    train_dl, val_dl, test_dl = get_dataloaders()
    model, best_loss, epochs = load_model()
    optim = get_optim(model)
    val_stats = val_epoch(val_dl,model)
    while True:
        train_stats = train_epoch(train_dl,model,optim)
        for key in train_stats:
            _run.log_scalar(f'training.{key}', train_stats[key])

        val_stats = val_epoch(val_dl,model)
        for key in val_stats:
            _run.log_scalar(f'validation.{key}', val_stats[key])

        if val_stats['loss'] < best_loss:
            best_loss = val_stats['loss']
            save_models(model,val_stats,epochs)
        epochs += 1
