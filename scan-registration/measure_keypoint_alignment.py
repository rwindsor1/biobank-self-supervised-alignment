'''
Code to register the two DXA and MRI datasets

'''
import sys, os, glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
sys.path.append('/users/rhydian/self-supervised-project')

from keypoint_alignment_dataset import KeypointAlignmentDataset
from models.SSECEncoders import VGGEncoder, HighResVGGEncoder
from train_SSEC import load_models
import train_angle_regressor

from registration import LowesTest, get_SIFT_correspondance, get_dense_correspondance, transform_points, find_best_rigid, RefinementNetwork
from plotting import show_matches, show_warped_perspective, show_transformed_keypoints
import torch.nn.functional as F

# configuration for alignment
class config():
    TRAINING_AUGMENTATION=False
    DATASET_ROOT = '/scratch/shared/beegfs/rhydian/UKBiobank'
    #MODEL_LOAD_PATH = '/users/rhydian/self-supervised-project/model_weights/SSECEncodersLowTemp5e-3'
    MODEL_LOAD_PATH = '/users/rhydian/self-supervised-project/model_weights/SSECEncodersBothBoth'
    #MODEL_LOAD_PATH = '/users/rhydian/self-supervised-project/model_weights/SSECEncodersHighRes2'
    USE_EQUAL_RES = True
    # ENCODER_TYPE = HighResVGGEncoder
    ENCODER_TYPE = VGGEncoder
    EMBEDDING_SIZE = 128
    USE_CUDA = False
    UPSAMPLE_FACTOR = 2 # amount feature maps scaled up for subpixel accuracy
    USE_CYCLIC = True # use cyclic point correlation
    USE_RANSAC = True
    RANSAC_TOLERANCE = 20
    GRID_RES = 1
    ALLOW_SCALING = False
    NUM_DISCRIM_POINTS = int(200*GRID_RES**2) # number of correlating pairs to find
    METHOD = 'DENSE' # from DENSE, SIFT, MATCH_KEYPOINT
    ROTATION_RANGE = (-2,2)
    ROTATION_SAMPLES=40
    LOWES_RATIO=0.92
    REFINEMENT_NETWORK_PATH='../model_weights/angle_regressor4'
    assert METHOD in ['MATCH_KEYPOINT','DENSE','REFINEMENT', 'SIFT','NONE']


def set_up():
    global c
    ds  = KeypointAlignmentDataset(reload_if_fail=False)
    dxa_model, mri_model, val_stats, epochs = load_models(c.ENCODER_TYPE, c.EMBEDDING_SIZE,
                                                          c.MODEL_LOAD_PATH, c.USE_CUDA, SINGLE_MRI=0, SINGLE_DXA=0 )
    refinement_model, _, _ = train_angle_regressor.load_model(c.REFINEMENT_NETWORK_PATH, c.USE_CUDA)
    if c.USE_CUDA:
        refinement_model.cuda()
    return ds, dxa_model, mri_model, refinement_model, val_stats, epochs


def save_registration_examples(ds, dxa_model, mri_model, refinement_model):
    global c
    all_errors = []
    all_best_poss_errors = []
    pbar = tqdm(range(len(ds)))
    labels=['left_arm', 'right_arm','spine_base','left_leg','right_leg']
    best_scalings = []
    times = []
    bad_idxes = [19,20,34,53]
    for idx in pbar:
        if idx in bad_idxes: continue
        tgt_pixel_spacing = 0.24
        sample = ds[idx]
        if c.USE_EQUAL_RES:
            #sample['dxa_img'] = F.interpolate(sample['dxa_img'][None], scale_factor=1/0.911)[0]
            #sample['target'] = F.interpolate(sample['target'][None], scale_factor=1/0.911)[0]
            tgt_pixel_spacing = tgt_pixel_spacing*0.911
            sample['dxa_keypoints'] = ((np.array(sample['dxa_keypoints'])*1/0.911 + np.array([10,0]))).tolist()


        tgt_img = sample['dxa_img']; tgt_model = dxa_model; tgt_pts = sample['dxa_keypoints']
        src_img = sample['mri_img']; src_model = mri_model; src_pts = sample['mri_keypoints']
        tick = time.time()
        R=None
        if c.METHOD == 'MATCH_KEYPOINT':
#             src_inliers, tgt_inliers,M,_,_,_,error = get_matching_keypoints_correspondance(
#                                                             src_img,tgt_img,src_model, tgt_model,
#                                                             c.USE_CUDA, c.UPSAMPLE_FACTOR, c.USE_CYCLIC,c.NUM_DISCRIM_POINTS,
#                                                             c.USE_RANSAC, c.RANSAC_TOLERANCE, c.GRID_RES, c.ALLOW_SCALING)
            good_matches,M,_,_ = LowesTest(sample['dxa_img'][None], sample['mri_img'][None], dxa_model, mri_model,
                                       c.LOWES_RATIO,True,c.RANSAC_TOLERANCE,c.USE_CUDA)
        elif c.METHOD == 'DENSE':
            M,best_idx,angles = get_dense_correspondance(src_img,tgt_img,src_model,tgt_model,
                                                         c.USE_CUDA, c.UPSAMPLE_FACTOR,c.ROTATION_RANGE, c.ROTATION_SAMPLES)

        elif c.METHOD=='SIFT':
            raise NotImplementedError()
        elif c.METHOD=='REFINEMENT':
            M,R,_,_,_,_,_,_= RefinementNetwork(sample['dxa_img'][None], sample['mri_img'][None], dxa_model, mri_model,c.LOWES_RATIO,c.RANSAC_TOLERANCE,c.USE_CUDA, refinement_model)
            for idx in range(len(src_pts)):
                src_pts[idx] =(R@np.array(src_pts[idx]+[1])).tolist()


        elif c.METHOD=='NONE':
            M = np.eye(3)

        tock=time.time()
        times.append(tock-tick)

        est_tgt_pts = transform_points(src_pts,M)
        errors = np.linalg.norm(tgt_pts - est_tgt_pts, axis=1)*tgt_pixel_spacing
        if np.mean(errors)>3:
            plt.subplot(121)
            plt.imshow(sample['mri_img'][0])
            for pt in src_pts:
                plt.scatter(pt[1],pt[0])
            plt.subplot(122)
            plt.imshow(sample['dxa_img'][0])
            for pt in est_tgt_pts:
                plt.scatter(pt[1],pt[0])
            for pt in tgt_pts:
                plt.scatter(pt[1],pt[0],marker='x')
            plt.title(f'Error: {np.mean(errors):.4}')
            plt.savefig(f'failure_cases/failure_{c.METHOD}_{idx}.png')
            plt.close('all')

        all_errors.append(errors)
        best_poss_M, best_scaling, best_angle, best_t = find_best_rigid(src_pts, tgt_pts)
        best_scalings.append(best_scaling)

        best_poss_est_tgt_pts = transform_points(src_pts, best_poss_M)
        best_poss_errors = np.linalg.norm(tgt_pts - best_poss_est_tgt_pts, axis=1)*tgt_pixel_spacing
        all_best_poss_errors.append(best_poss_errors)

        pbar.set_description(f'Err: {np.mean(all_errors):.4} +- {np.std(all_errors):.4} Med {np.median(all_errors):.4}, Best Poss: {np.mean(all_best_poss_errors):.4} +- {np.std(all_best_poss_errors):.4}')
        fig, ax = show_transformed_keypoints(src_img, tgt_img, src_pts, est_tgt_pts, tgt_pts)
        fig.savefig(f'../images/keypoint_aligns_equal_res/keypoints_{idx}.png')
        plt.close('all')
        #fig, ax = show_matches(src_img, tgt_img,src_inliers, tgt_inliers)
        #fig.tight_layout()
        #fig.savefig(f'../images/keypoint_aligns_equal_res/keypoint_matches/example_keypoint_match_{idx}.png')

    pbar.close()
    all_best_poss_errors = np.array(all_best_poss_errors)
    all_errors = np.array(all_errors)
    for idx, label in enumerate(labels):
        print(f"{label}: {np.mean(all_errors[:,idx]):.4} +- {np.std(all_errors[:,idx]):.4}, Median {np.median(all_errors[:,idx]):.4}")

    arms = np.concatenate([all_errors[:,0],all_errors[:,1]],axis=0)
    legs = np.concatenate([all_errors[:,-2],all_errors[:,-1]],axis=0)
    print(f'Arms: {np.mean(arms):.4} +- {np.std(arms):.4}, Median {np.median(arms):.4}')
    print(f'Legs: {np.mean(legs):.4} +- {np.std(legs):.4}, Median {np.median(legs):.4}')

    print(f"Time: {np.mean(times)} +- {np.std(times)}")
    print('--------------------------')
    print('Best Possible')
    for idx, label in enumerate(labels):
        print(idx)
        print(f"{label}: {np.mean(all_best_poss_errors[:,idx]):.4} +- {np.std(all_best_poss_errors[:,idx]):.4}, Median +-{np.median(all_best_poss_errors[:,idx]):.4}")
    arms = np.concatenate([all_best_poss_errors[:,0], all_best_poss_errors[:,1]],axis=0)
    legs = np.concatenate([all_best_poss_errors[:,-2],all_best_poss_errors[:,-1]],axis=0)
    print(f'Arms: {np.mean(arms):.4} +- {np.std(arms):.4}, Median {np.median(arms):.4}')
    print(f'Legs: {np.mean(legs):.4} +- {np.std(legs):.4}, Median {np.median(legs):.4}')
    return


c = config()
if __name__ == '__main__':
    ds, dxa_model, mri_model, refinement_model, val_stats, epochs = set_up()
    save_registration_examples(ds, dxa_model, mri_model, refinement_model)

    import pdb; pdb.set_trace()


