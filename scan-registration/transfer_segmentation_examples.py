'''
Code to register the two DXA and MRI datasets

'''
import sys, os, glob
from tqdm import tqdm
import cv2
import numpy as np
sys.path.append('/users/rhydian/self-supervised-project')

from datasets.BothMidCoronalDataset import BothMidCoronalDataset
from models.SSECEncoders import VGGEncoder, HighResVGGEncoder
from train_SSEC import load_models
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchgeometry as tgm

from registration import get_matching_keypoints
from plotting import show_matches, show_warped_perspective, show_transformed_segmentations

# configuration for alignment
class config():
    TRAINING_AUGMENTATION=False
    DATASET_ROOT = '/scratch/shared/beegfs/rhydian/UKBiobank'
    MODEL_LOAD_PATH = '/users/rhydian/self-supervised-project/model_weights/SSECEncodersEqualResLowTemp'
    USE_EQUAL_RES = True
    #ENCODER_TYPE = HighResVGGEncoder
    ENCODER_TYPE = VGGEncoder
    EMBEDDING_SIZE = 128
    USE_CUDA = True
    UPSAMPLE_FACTOR = 2 # amount feature maps scaled up for subpixel accuracy
    USE_CYCLIC = True # use cyclic point correlation
    USE_RANSAC = True
    RANSAC_TOLERANCE = 15
    GRID_RES = 1
    ALLOW_SCALING = False
    NUM_DISCRIM_POINTS = int(200*GRID_RES**2) # number of correlating pairs to find


def set_up():
    global c
    train_ds = BothMidCoronalDataset(set_type='train', all_root=c.DATASET_ROOT, return_segmentations=True, augment=c.TRAINING_AUGMENTATION, equal_res=False)
    val_ds   = BothMidCoronalDataset(set_type='val'  , all_root=c.DATASET_ROOT, return_segmentations=True, augment=False, equal_res=False)
    test_ds  = BothMidCoronalDataset(set_type='test' , all_root=c.DATASET_ROOT, return_segmentations=True, augment=False, equal_res=False)
    dxa_model, mri_model, val_stats, epochs = load_models(c.ENCODER_TYPE, c.EMBEDDING_SIZE, True, c.MODEL_LOAD_PATH, c.USE_CUDA )
    return train_ds, val_ds, test_ds, dxa_model, mri_model, val_stats, epochs


def save_registration_examples(ds, dxa_model, mri_model):
    global c
    for idx in tqdm(range(100)):
        sample = ds[idx]
        mri_img = sample['mri_img']; mri_model = mri_model;
        dxa_img = sample['dxa_img']; dxa_model = dxa_model
        dxa_seg = sample['target']

        dxa_inliers, mri_inliers,M, scaling, angle, t, error = get_matching_keypoints(dxa_img,mri_img,dxa_model, mri_model,
                                                          c.USE_CUDA, c.UPSAMPLE_FACTOR, c.USE_CYCLIC,c.NUM_DISCRIM_POINTS,
                                                          c.USE_RANSAC, c.RANSAC_TOLERANCE, c.GRID_RES, c.ALLOW_SCALING)
        # angle=0

        pred_mri_seg = dxa_seg
        pred_mri_seg = F.pad(pred_mri_seg,(int(t[1]),0,int(t[0]),0))
        hdiff = mri_img.shape[-2] - pred_mri_seg.shape[-2]
        wdiff = mri_img.shape[-1] - pred_mri_seg.shape[-1]
        pred_mri_seg = F.pad(pred_mri_seg, (0,wdiff,0,hdiff))
        pred_mri_seg = TF.rotate(pred_mri_seg,angle,center=(0,0))
        # pad to right shape
        fig, axs = show_transformed_segmentations(mri_img, dxa_img, pred_mri_seg, dxa_seg, np.linalg.inv(M))
        axs[1].set_title(f'{angle:.4}, error: {error:.4}')
        fig.savefig(f'../images/segmentation_transfer2/example_{idx}')




c = config()
if __name__ == '__main__':
    train_ds, val_ds, test_ds, dxa_model, mri_model, val_stats, epochs = set_up()
    dxa_model.eval()
    mri_model.eval()
    save_registration_examples(test_ds, dxa_model, mri_model)



