'''
Code to register the two DXA and MRI datasets

'''
import sys, os, glob
from tqdm import tqdm
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
sys.path.append('/users/rhydian/self-supervised-project')

from datasets.BothMidCoronalDataset import BothMidCoronalDataset
from models.SSECEncoders import VGGEncoder, HighResVGGEncoder
from gen_utils import *
from train_SSEC import load_models
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchgeometry as tgm
import torch
import train_angle_regressor

from registration import get_dense_correspondance, RefinementNetwork
from plotting import show_matches, show_warped_perspective, show_transformed_segmentations

# configuration for alignment
class config():
    TRAINING_AUGMENTATION=False
    DATASET_ROOT = '/scratch/shared/beegfs/rhydian/UKBiobank'
    MODEL_LOAD_PATH = '/users/rhydian/self-supervised-project/model_weights/SSECEncodersBothBoth'
    USE_EQUAL_RES = True
    #ENCODER_TYPE = HighResVGGEncoder
    ENCODER_TYPE = VGGEncoder
    EMBEDDING_SIZE = 128
    USE_CUDA = False
    UPSAMPLE_FACTOR = 2 # amount feature maps scaled up for subpixel accuracy
    USE_CYCLIC = True # use cyclic point correlation
    USE_RANSAC = True
    RANSAC_TOLERANCE = 15
    GRID_RES = 1
    ALLOW_SCALING = False
    NUM_DISCRIM_POINTS = int(200*GRID_RES**2) # number of correlating pairs to find
    SAVE_SEGMENTATIONS_PATH = '/scratch/shared/beegfs/rhydian/UKBiobank/TransferredMRISegmentations2'
    WARPED_DXA_PATH = '/scratch/shared/beegfs/rhydian/UKBiobank/WarpedDXAs'
    ERROR_SAVE_PATH = 'segmentation_save_errors.txt'
    ROTATION_RANGE = (-2,2)
    LOWES_RATIO=0.93
    REFINEMENT_NETWORK_PATH='../model_weights/angle_regressor4'
    ROTATION_SAMPLES=40
    OUT_RELATIVE_TRANSFORM_FILE='scan_relative_transforms.csv'

def set_up(set_type):
    global c
    ds = BothMidCoronalDataset(set_type=set_type, all_root=c.DATASET_ROOT, return_segmentations=True, return_scan_info=True, augment=c.TRAINING_AUGMENTATION, equal_res=True)
    dxa_model, mri_model, val_stats, epochs = load_models(c.ENCODER_TYPE, c.EMBEDDING_SIZE, c.MODEL_LOAD_PATH, c.USE_CUDA,SINGLE_MRI=False,SINGLE_DXA=False)
    refinement_model, _, _ = train_angle_regressor.load_model(c.REFINEMENT_NETWORK_PATH, c.USE_CUDA)
    if c.USE_CUDA:
        refinement_model.cuda()
    return ds, dxa_model, mri_model, refinement_model, val_stats, epochs

def get_mri_segmentation(mri_img, mri_model, dxa_img, dxa_model, dxa_seg, refinement_model):
    global c
    # dxa_inliers, mri_inliers,M, scaling, angle, t, error = get_matching_keypoints(dxa_img,mri_img,dxa_model, mri_model,
    #                                                   c.USE_CUDA, c.UPSAMPLE_FACTOR, c.USE_CYCLIC,c.NUM_DISCRIM_POINTS,
    #                                                   c.USE_RANSAC, c.RANSAC_TOLERANCE, c.GRID_RES, c.ALLOW_SCALING)

    M,R, angle, t_x,t_y, coarse_angle, coarse_t_x, coarse_t_y = RefinementNetwork(dxa_img[None], mri_img[None], dxa_model, mri_model,
                            c.LOWES_RATIO,c.RANSAC_TOLERANCE,c.USE_CUDA, refinement_model)
    new_M = M.copy(); new_M[1,2],new_M[0,2]=-new_M[0,2],-new_M[1,2]; 
    new_M=np.linalg.inv(M); new_M[0,1],new_M[1,0]=new_M[1,0],new_M[0,1]; new_M[1,2],new_M[0,2]=new_M[0,2],new_M[1,2]
    warped_dxa = cv2.warpPerspective(np.array(255*dxa_img.permute(1,2,0)).astype('uint8'),
                                     new_M,(mri_img.shape[-1], mri_img.shape[-2]))
    warped_dxa = torch.Tensor(warped_dxa).permute(2,0,1).float()[None]/255
    warped_dxa = TF.affine(warped_dxa, -angle, [-t_x,-t_y],1,[0,0])[0]
    warped_dxa_seg = cv2.warpPerspective(np.array(dxa_seg).transpose(1,2,0).astype('uint8'),new_M,(mri_img.shape[-1], mri_img.shape[-2]))
    warped_dxa_seg = torch.Tensor(warped_dxa_seg).permute(2,0,1)
    warped_dxa_seg = TF.affine(torch.Tensor(warped_dxa_seg[None]), -angle, [-t_x,-t_y],1,[0,0])[0]
    new_dxa = TF.affine(TF.rotate(dxa_img[None],coarse_angle,center=(0,0)),0,(coarse_t_y,coarse_t_x),1,(0,0))[:,:,:mri_img.shape[-2],:mri_img.shape[-1]]
    plt.imshow(grayscale(mri_img[0])+red(new_dxa[0,0]))


    return warped_dxa_seg, warped_dxa,M, coarse_angle, coarse_t_x, coarse_t_y

def transfer_and_save_segmentations(ds, dxa_model, mri_model,start_idx, end_idx, refinement_model):
    global c
    for idx in tqdm(range(start_idx, end_idx)):
        sample = ds[idx]
        mri_img = sample['mri_img']; mri_model = mri_model;
        dxa_img = sample['dxa_img']; dxa_model = dxa_model
        dxa_seg = sample['target']
        try:
            pred_mri_seg, warped_dxa, M, angle, t_x, t_y = get_mri_segmentation(mri_img, mri_model,dxa_img,dxa_model,dxa_seg, refinement_model)
            print(','.join([str(x) for x in [sample['mri_filename'], sample['dxa_filename'], angle, t_x, t_y]])+'\n')
            with open(c.OUT_RELATIVE_TRANSFORM_FILE,'a') as f:
                f.write(','.join([str(x) for x in [sample['mri_filename'], sample['dxa_filename'], angle, t_x, t_y]])+'\n')

        except Exception as E:
            sample = ds[idx]
            with open(c.ERROR_SAVE_PATH,'w') as f:
                f.write(f"{sample['mri_filename']}, {E}\n")
            print(E)
            print('failed')

parser = argparse.ArgumentParser()
parser.add_argument('--start_frac', type=float, help='The fraction of the dataset to start at')
parser.add_argument('--end_frac',   type=float, help='The fraction of the dataset to end at')
parser.add_argument('--set_type',   type=str, help='The dataset to run over')
args = parser.parse_args()
c = config()
if __name__ == '__main__':
    assert args.set_type in ['train','test','val']
    ds, dxa_model, mri_model, refinement_model, val_stats, epochs = set_up(args.set_type)
    dxa_model.eval()
    mri_model.eval()
    start_idx = int(np.floor(len(ds)*args.start_frac))
    end_idx = int(np.floor(len(ds)*args.end_frac))
    transfer_and_save_segmentations(ds, dxa_model, mri_model,start_idx,end_idx, refinement_model)



