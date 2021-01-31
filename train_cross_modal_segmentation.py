import sys,os,glob
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

from models.SpatialVGGM import SpatialVGGM
from models.SegmentationUnetLowRes import SegmentationUnetLowRes
from datasets.SegmentationDatasets import MRISlicesSegmentationDataset
from gen_utils import balanced_l1w_loss, grayscale, red, dice_and_iou
from loss_functions import ContrastiveLoss
from sacred import Experiment
from sacred.observers import MongoObserver
# to make sensible stop
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'

ex = Experiment('CrossModalSegmentation')
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
ex.observers.append(MongoObserver(url='login1.triton.cluster:27017', db_name='SelfSupervisedProject'))

@ex.config
def expt_config():
    LR = 0.001
    ADAM_BETAS = (0.9,0.999)
    NUM_WORKERS = 20
    TRAIN_NUM_WORKERS = NUM_WORKERS
    VAL_NUM_WORKERS   = NUM_WORKERS
    TEST_NUM_WORKERS  = NUM_WORKERS
    BATCH_SIZE = 20
    TRAIN_BATCH_SIZE = BATCH_SIZE
    VAL_BATCH_SIZE = BATCH_SIZE
    TEST_BATCH_SIZE = BATCH_SIZE
    TRAINING_ITERATIONS = 20000/BATCH_SIZE 
    MARGIN = 1
    VALIDATION_ITERATIONS = 100
    POOLING_STRATEGY = 'max'
    assert POOLING_STRATEGY in ['max', 'average']
    BLANKING_AUGMENTATION = False
    USE_INSTANCE_LOSS = True
    LOAD_WEIGHTS_PATH = 'fully_trained_model_weights/AvgPool'
    SAVE_WEIGHTS_PATH = 'model_weights/AvgPoolUnet'
    SAVE_IMAGES_PATH = 'images/contrastive_examples' # the path to save responses of the network from the save_examples command
    NUM_SAVE_IMAGES = 5 # how many images to save when running `save_examples`
    DATASET_ROOT = '/tmp/rhydian/'

@ex.capture
def load_embeddings_models(LOAD_WEIGHTS_PATH, POOLING_STRATEGY, use_cuda=True):
    dxa_model = SpatialVGGM(pooling=POOLING_STRATEGY)
    mri_model = SpatialVGGM(pooling=POOLING_STRATEGY)
    dxa_model = nn.DataParallel(dxa_model)
    mri_model = nn.DataParallel(mri_model)
    dxa_model_dict = dxa_model.state_dict()
    mri_model_dict = mri_model.state_dict()
    print(f'Trying to load from {LOAD_WEIGHTS_PATH}')
    # load model weights
    if os.path.isdir(LOAD_WEIGHTS_PATH):
            list_of_pt = glob.glob(LOAD_WEIGHTS_PATH + '/*.pt')
            if len(list_of_pt):
                latest_pt = max(list_of_pt, key=os.path.getctime)
                checkpoint = torch.load(latest_pt, map_location=torch.device('cpu'))
                dxa_model_dict.update(checkpoint['dxa_model_weights'])
                mri_model_dict.update(checkpoint['mri_model_weights'])
                dxa_model.load_state_dict(dxa_model_dict)
                mri_model.load_state_dict(mri_model_dict)
                best_loss = checkpoint['loss']
                epochs = checkpoint['epochs'] 
                print(f"==> Resuming model trained for {epochs} epochs...")
            else:
                print("==> Training Fresh Model ")
                best_loss = 99999999999999
                epochs = 0
    else:
        raise Exception(f'Could not find directory at {LOAD_WEIGHTS_PATH}')
        
    if use_cuda:
        dxa_model.cuda()
        mri_model.cuda()
    return dxa_model, mri_model, best_loss, epochs

@ex.capture
def load_segmentation_unet(encoding_model):
    unet = SegmentationUnetLowRes(encoding_model, use_skips=False)
    for parameter in unet.SpatialEncoder.parameters(): parameter.requires_grad = False
    return unet

@ex.capture
def get_dataloaders(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE, TRAIN_NUM_WORKERS, VAL_NUM_WORKERS, TEST_NUM_WORKERS, DATASET_ROOT):
    train_ds = MRISlicesSegmentationDataset(set_type='train', all_root=DATASET_ROOT, augment=True)
    val_ds   = MRISlicesSegmentationDataset(set_type='val'  , all_root=DATASET_ROOT, augment=False)
    test_ds  = MRISlicesSegmentationDataset(set_type='test' , all_root=DATASET_ROOT, augment=False)
    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=False, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=VAL_BATCH_SIZE,   num_workers=VAL_NUM_WORKERS,   shuffle=False, drop_last=True)
    test_dl  = DataLoader(test_ds,  batch_size=TEST_BATCH_SIZE,  num_workers=TEST_NUM_WORKERS,  shuffle=False, drop_last=True)
    return train_dl, val_dl, test_dl

@ex.capture
def get_optimizers(unet, LR, ADAM_BETAS):
    optim = Adam(unet.parameters(), lr=LR, betas=ADAM_BETAS)
    return optim

@ex.capture
def train_epoch(dl, unet, optim, _run):
    pbar = tqdm(dl)
    criterion = nn.BCEWithLogitsLoss()
    unet.cuda()
    unet.train()
    losses = []
    dices = []
    for idx, sample in enumerate(pbar):
        optim.zero_grad()
        predictions = unet(sample['dxa_vol'].cuda())
        target      = sample['dxa_segmentation'].cuda()
        l_spine     = criterion(predictions[:,0], target[:,1].float())
        l_spine.backward()
        optim.step()
        dice, iou = dice_and_iou(predictions[:,0], target[:,1].float())
        losses.append(l_spine.item())
        dices.append(dice)
        pbar.set_description(f'Loss: {l_spine.item():.3}, Dice: {dice:.3}')

    avg_loss = np.mean(losses)
    avg_dice = np.mean(dices)
    print(f"Avg. Loss: {avg_loss:.3}, Avg. Dice: {avg_dice:.3}")
    _run.log_scalar('training.loss', avg_loss)
    _run.log_scalar('training.dice', avg_dice)
    return avg_loss

@ex.capture
def val_epoch(dl, unet, optim, _run):
    pbar = tqdm(dl)
    criterion = nn.BCEWithLogitsLoss()
    unet.cuda()
    unet.eval()
    losses = []
    dices = []
    for idx, sample in enumerate(pbar):
        optim.zero_grad()
        with torch.no_grad():
            predictions = unet(sample['dxa_vol'].cuda())
        target      = sample['dxa_segmentation'].cuda()
        l_spine     = criterion(predictions[:,0], target[:,1].float())
        dice, iou = dice_and_iou(predictions[:,0], target[:,1].float())
        losses.append(l_spine.item())
        dices.append(dice)
        pbar.set_description(f'Loss: {l_spine.item():.3}, Dice: {dice:.3}')

    avg_loss = np.mean(losses)
    avg_dice = np.mean(dices)
    print(f"Avg. Loss: {avg_loss:.3}, Avg. Dice: {avg_dice:.3}")
    _run.log_scalar('validation.loss', avg_loss)
    _run.log_scalar('validation.dice', avg_dice)
    return avg_loss

@ex.capture
def save_model(unet, loss, epochs, SAVE_WEIGHTS_PATH):
    print(f'==> Saving Model Weights to {SAVE_WEIGHTS_PATH}')
    state = {'unet_model_weights': unet.state_dict(),
             'loss'             : loss,
             'epochs'           : epochs
            }
    if not os.path.isdir(SAVE_WEIGHTS_PATH):
        os.mkdir(SAVE_WEIGHTS_PATH)
    previous_checkpoints = glob.glob(SAVE_WEIGHTS_PATH + '/ckpt*.pt', recursive=True)
    for previous_checkpoint in previous_checkpoints:
       os.remove(previous_checkpoint)
    torch.save(state, SAVE_WEIGHTS_PATH + '/ckpt' + str(epochs) + '.pt')
    return

@ex.automain
def main():
    dxa_model, mri_model, best_loss, epoch_no = load_embeddings_models(use_cuda=False)
    unet = load_segmentation_unet(dxa_model)
train_dl, val_dl, test_dl = get_dataloaders()
    optim  = get_optimizers(unet)
    epochs = 0
    best_loss = 99999999
    while True:
        unet.SpatialEncoder = dxa_model
        for parameter in unet.SpatialEncoder.parameters(): parameter.requires_grad = False
        print(f'Training, Epoch {epochs}...')
        train_loss = train_epoch(train_dl, unet, optim)
        print(f'Validating, Epoch {epochs}...')
        val_loss = val_epoch(val_dl, unet, optim)
        if val_loss < best_loss:
            save_model(unet, val_loss, epochs)
            # TODO
            pass
        epochs += 1