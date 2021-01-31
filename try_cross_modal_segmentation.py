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
from copy import deepcopy 

from models.SpatialVGGM import SpatialVGGM
from models.SegmentationUnetLowRes import SegmentationUnetLowRes
from datasets.SegmentationDatasets import MRISlicesSegmentationDataset
from gen_utils import balanced_l1w_loss, grayscale, red, dice_and_iou
from loss_functions import ContrastiveLoss
from train_cross_modal_segmentation import load_embeddings_models


class config():
    POOLING_STRATEGY = 'average'
    UNET_PATH = 'model_weights/AvgPoolUnet'
    EMBEDDINGS_PATH = 'fully_trained_model_weights/AvgPool'

def load_segmentation_unet(UNET_PATH, POOLING_STRATEGY):
    unet = SegmentationUnetLowRes(nn.DataParallel(SpatialVGGM(pooling=POOLING_STRATEGY)), use_skips=False)
    unet_model_dict = unet.state_dict()
    if os.path.isdir(UNET_PATH):
            list_of_pt = glob.glob(UNET_PATH + '/*.pt')
            if len(list_of_pt):
                latest_pt = max(list_of_pt, key=os.path.getctime)
                checkpoint = torch.load(latest_pt, map_location=torch.device('cpu'))
                unet_model_dict.update(checkpoint['unet_model_weights'])
                unet.load_state_dict(unet_model_dict)
                best_loss = checkpoint['loss']
                epochs = checkpoint['epochs'] 
                print(f"==> Resuming model trained for {epochs} epochs...")
            else:
                print("==> Training Fresh Model ")
                best_loss = 99999999999999
                epochs = 0
    else:
        raise Exception(f'Could not find directory at {LOAD_WEIGHTS_PATH}')

    return unet


dxa_model, mri_model, best_loss, epoch_no = load_embeddings_models(config.EMBEDDINGS_PATH, config.POOLING_STRATEGY, use_cuda=False)
unet = load_segmentation_unet(config.UNET_PATH, config.POOLING_STRATEGY)
unet.eval()


test_ds  = MRISlicesSegmentationDataset(set_type='test' , all_root='/scratch/shared/beegfs/rhydian/UKBiobank/', augment=False)

for idx, sample in enumerate(test_ds):
    dxa_encoder_unet = deepcopy(unet)
    mri_encoder_unet = deepcopy(unet)
    mri_encoder_unet.SpatialEncoder = mri_model
    dxa_encoder_unet.SpatialEncoder = dxa_model
    for unet in [mri_encoder_unet, dxa_encoder_unet]:
        for parameter in unet.SpatialEncoder.parameters(): parameter.requires_grad = False
    mri_encoder_unet.eval()
    dxa_encoder_unet.eval()
    with torch.no_grad():
        mri_predictions = mri_encoder_unet(sample['mri_vol'][None])
        dxa_predictions = dxa_encoder_unet(sample['dxa_vol'][None])

    mri_embeddings = mri_encoder_unet.get_transformed_spatial_embeddings(sample['mri_vol'][None])
    dxa_embeddings = dxa_encoder_unet.get_transformed_spatial_embeddings(sample['dxa_vol'][None])
    import pdb; pdb.set_trace()

    # plt.imshow(grayscale(sample['mri_vol'][0].numpy())+red(mri_predictions[1,0].numpy()))
    plt.imshow(grayscale(sample['dxa_vol'][0].numpy())+red(dxa_predictions[0,0].numpy()))

    plt.show()