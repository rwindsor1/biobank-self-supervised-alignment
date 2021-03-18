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

from registration import *
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
    UPSAMPLE_FACTOR =2 # amount feature maps scaled up for subpixel accuracy
    USE_CYCLIC = True # use cyclic point correlation
    USE_RANSAC = True
    RANSAC_TOLERANCE = 15
    GRID_RES = 1
    ALLOW_SCALING = False
    NUM_DISCRIM_POINTS = int(200*GRID_RES**2) # number of correlating pairs to find
    SAVE_SEGMENTATIONS_PATH = '/scratch/shared/beegfs/rhydian/UKBiobank/TransferredMRISegmentations'
    OUT_PATH = '/scratch/shared/beegfs/rhydian/UKBiobank/registration_refinement_images'
    ERROR_SAVE_PATH = 'segmentation_save_errors.txt'
    LOWES_RATIO=0.93
    ROTATION_RANGE = (-2,2)
    ROTATION_SAMPLES=40

def set_up(set_type):
    global c
    ds = BothMidCoronalDataset(set_type=set_type, all_root=c.DATASET_ROOT, return_segmentations=True, return_scan_info=True, augment=c.TRAINING_AUGMENTATION, equal_res=True)
    dxa_model, mri_model, val_stats, epochs = load_models(c.ENCODER_TYPE, c.EMBEDDING_SIZE, c.MODEL_LOAD_PATH, c.USE_CUDA,SINGLE_MRI=False,SINGLE_DXA=False)
    return ds, dxa_model, mri_model, val_stats, epochs

def get_mri_segmentation(mri_img, mri_model, dxa_img, dxa_model, dxa_seg):
    global c
    # dxa_inliers, mri_inliers,M, scaling, angle, t, error = get_matching_keypoints(dxa_img,mri_img,dxa_model, mri_model,
    #                                                   c.USE_CUDA, c.UPSAMPLE_FACTOR, c.USE_CYCLIC,c.NUM_DISCRIM_POINTS,
    #                                                   c.USE_RANSAC, c.RANSAC_TOLERANCE, c.GRID_RES, c.ALLOW_SCALING)

    M,best_idx,angles = get_dense_correspondance(dxa_img,mri_img,dxa_model,mri_model,
                                                 c.USE_CUDA, c.UPSAMPLE_FACTOR,c.ROTATION_RANGE, c.ROTATION_SAMPLES)

    # good_matches,M = LowesTest(dxa_img[None], mri_img[None], dxa_model, mri_model,
    #                            c.LOWES_RATIO,True,c.RANSAC_TOLERANCE,c.USE_CUDA)
    new_M = M.copy(); new_M[1,2],new_M[0,2]=-new_M[0,2],-new_M[1,2];
    warped_dxa = cv2.warpPerspective(np.array(255*dxa_img.permute(1,2,0)).astype('uint8'),new_M,(dxa_img.shape[-1], dxa_img.shape[-2])).astype(float)/255;
    warped_dxa_seg = cv2.warpPerspective(np.array(dxa_seg).transpose(1,2,0).astype('uint8'),new_M,(dxa_img.shape[-1], mri_img.shape[-2])).astype(bool)
    return warped_dxa_seg, warped_dxa,M

def transfer_and_save_segmentations(ds, dxa_model, mri_model,start_idx, end_idx, set_type):
    global c
    for idx in tqdm(range(start_idx, end_idx)):
        try:
            sample = ds[idx]
            mri_img = sample['mri_img']; mri_model = mri_model;
            dxa_img = sample['dxa_img']; dxa_model = dxa_model
            mri_model.eval()
            dxa_model.eval()
            dxa_seg = sample['target']

            pred_mri_seg, warped_dxa, M = get_mri_segmentation(mri_img, mri_model,dxa_img,dxa_model,dxa_seg)
            #plt.subplot(131)
            #plt.imshow(sample['mri_img'][0])
            #plt.subplot(132)
            #plt.imshow(warped_dxa[:,:,0])
            #plt.subplot(132)
            #plt.imshow(red(warped_dxa[:sample['mri_img'].shape[-2],:sample['mri_img'].shape[-1],0])+grayscale(sample['mri_img'][0]))
            #plt.savefig('test.png')
            #warped_dxa = torch.Tensor(warped_dxa).permute(2,0,1)
            #os.system('imgcat test.png')
            #print(warped_dxa.shape)

            #imgs = torch.cat([warped_dxa, torch.Tensor(mri_img)],axis=0)
            imgs = [torch.Tensor(warped_dxa),torch.Tensor(mri_img)]
            torch.save(imgs, os.path.join(c.OUT_PATH, set_type, sample['mri_filename']+'.pt'))


            # torch.save(torch.Tensor(pred_mri_seg), os.path.join(c.SAVE_SEGMENTATIONS_PATH, f"{sample['mri_filename']}_2dsegmentation.pt"))
            # torch.save(torch.Tensor(warped_dxa), os.path.join(c.WARPED_DXA_PATH, f"{sample['mri_filename']}_warped_dxa.pt"))
        except Exception as E:
            sample = ds[idx]
            # with open(c.ERROR_SAVE_PATH,'w') as f:
            #     f.write(f"{sample['mri_filename']}, {E}\n")
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
    ds, dxa_model, mri_model, val_stats, epochs = set_up(args.set_type)
    dxa_model.eval()
    mri_model.eval()
    start_idx = int(np.floor(len(ds)*args.start_frac))
    end_idx = int(np.floor(len(ds)*args.end_frac))
    transfer_and_save_segmentations(ds, dxa_model, mri_model,start_idx,end_idx, args.set_type)



