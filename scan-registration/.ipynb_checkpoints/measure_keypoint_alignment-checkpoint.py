'''
Code to register the two DXA and MRI datasets

'''
import sys, os, glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/users/rhydian/self-supervised-project')

from keypoint_alignment_dataset import KeypointAlignmentDataset
from models.SSECEncoders import VGGEncoder, HighResVGGEncoder
from train_SSEC import load_models

from registration import get_matching_keypoints_correspondance, get_SIFT_correspondance, get_dense_correspondance, transform_points, find_best_rigid 
from plotting import show_matches, show_warped_perspective, show_transformed_keypoints
import torch.nn.functional as F

# configuration for alignment
class config():
    TRAINING_AUGMENTATION=False
    DATASET_ROOT = '/scratch/shared/beegfs/rhydian/UKBiobank'
    #MODEL_LOAD_PATH = '/users/rhydian/self-supervised-project/model_weights/SSECEncodersLowTemp5e-3'
    MODEL_LOAD_PATH = '/users/rhydian/self-supervised-project/model_weights/SSECEncodersBothBoth'
    USE_EQUAL_RES = True
    # ENCODER_TYPE = HighResVGGEncoder
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
    METHOD = 'MATCH_KEYPOINT' # from DENSE, SIFT, MATCH_KEYPOINT
    ROTATION_RANGE = (-2,2)
    ROTATION_SAMPLES=40
    
    assert METHOD in ['MATCH_KEYPOINT','DENSE','SIFT']


def set_up():
    global c
    ds  = KeypointAlignmentDataset()
    dxa_model, mri_model, val_stats, epochs = load_models(c.ENCODER_TYPE, c.EMBEDDING_SIZE, 
                                                          c.MODEL_LOAD_PATH, c.USE_CUDA, SINGLE_MRI=0, SINGLE_DXA=0 )
    return ds, dxa_model, mri_model, val_stats, epochs


def save_registration_examples(ds, dxa_model, mri_model):
    global c
    all_errors = []
    all_best_poss_errors = []
    pbar = tqdm(range(len(ds)))
    labels=['left_arm', 'right_arm','spine_base','left_leg','right_leg']
    best_scalings = []
    for idx in pbar:
        tgt_pixel_spacing = 0.24
        sample = ds[idx]
        if c.USE_EQUAL_RES:
            #sample['dxa_img'] = F.interpolate(sample['dxa_img'][None], scale_factor=1/0.911)[0]
            #sample['target'] = F.interpolate(sample['target'][None], scale_factor=1/0.911)[0]
            tgt_pixel_spacing = tgt_pixel_spacing*0.911
            sample['dxa_keypoints'] = ((np.array(sample['dxa_keypoints'])*1/0.911 + np.array([10,0]))).tolist()


        tgt_img = sample['dxa_img']; tgt_model = dxa_model; tgt_pts = sample['dxa_keypoints']
        src_img = sample['mri_img']; src_model = mri_model; src_pts = sample['mri_keypoints']
        if c.METHOD == 'MATCH_KEYPOINT':
#             src_inliers, tgt_inliers,M,_,_,_,error = get_matching_keypoints_correspondance(
#                                                             src_img,tgt_img,src_model, tgt_model,
#                                                             c.USE_CUDA, c.UPSAMPLE_FACTOR, c.USE_CYCLIC,c.NUM_DISCRIM_POINTS,
#                                                             c.USE_RANSAC, c.RANSAC_TOLERANCE, c.GRID_RES, c.ALLOW_SCALING)
            good_matches,M = LowesTest(sample['dxa_img'][None], sample['mri_img'][None], dxa_model, mri_model, 0.93,use_ransac=True)
        elif c.METHOD == 'DENSE':
            M,best_idx,angles = get_dense_correspondance(src_img,tgt_img,src_model,tgt_model,
                                                         c.USE_CUDA, c.UPSAMPLE_FACTOR,c.ROTATION_RANGE, c.ROTATION_SAMPLES) 

        elif c.METHOD=='SIFT':
            src_inliers, tgt_inliers,M = get_dataload_
            raise NotImplementedError()

        est_tgt_pts = transform_points(src_pts,M)
        errors = np.linalg.norm(tgt_pts - est_tgt_pts, axis=1)*tgt_pixel_spacing
        all_errors.append(errors)
        best_poss_M, best_scaling, best_angle, best_t = find_best_rigid(src_pts, tgt_pts)
        best_scalings.append(best_scaling)

        best_poss_est_tgt_pts = transform_points(src_pts, best_poss_M)
        best_poss_errors = np.linalg.norm(tgt_pts - best_poss_est_tgt_pts, axis=1)*tgt_pixel_spacing
        all_best_poss_errors.append(best_poss_errors)

        pbar.set_description(f'Err: {np.mean(all_errors):.4} +- {np.std(all_errors):.4}, Best Poss: {np.mean(all_best_poss_errors):.4} +- {np.std(all_best_poss_errors):.4}')
        fig, ax = show_transformed_keypoints(src_img, tgt_img, src_pts, est_tgt_pts, tgt_pts)
        fig.savefig(f'../images/keypoint_aligns_equal_res/keypoints_{idx}.png')
        plt.close('all')
        #fig, ax = show_matches(src_img, tgt_img,src_inliers, tgt_inliers)
        #fig.tight_layout()
        #fig.savefig(f'../images/keypoint_aligns_equal_res/keypoint_matches/example_keypoint_match_{idx}.png')

    pbar.close()
    all_errors = np.array(all_errors)
    for idx, label in enumerate(labels):
        print(f"{label}: {np.mean(all_errors[:,idx]):.4} +- {np.std(all_errors[:,idx]):.4}")
    return


c = config()
if __name__ == '__main__':
    ds, dxa_model, mri_model, val_stats, epochs = set_up()
    save_registration_examples(ds, dxa_model, mri_model)

    import pdb; pdb.set_trace()


