'''
Code to register the two DXA and MRI datasets

'''
import sys, os, glob
from tqdm import tqdm
import cv2
sys.path.append('/users/rhydian/self-supervised-project')

from datasets.BothMidCoronalDataset import BothMidCoronalDataset
from models.SSECEncoders import VGGEncoder, HighResVGGEncoder
from train_SSEC import load_models

from registration import get_matching_keypoints
from plotting import show_matches, show_warped_perspective

# configuration for alignment
class config():
    TRAINING_AUGMENTATION=False
    DATASET_ROOT = '/scratch/shared/beegfs/rhydian/UKBiobank'
    MODEL_LOAD_PATH = '/users/rhydian/self-supervised-project/model_weights/SSECEncodersEqualResLowTemp'
    #MODEL_LOAD_PATH = '/users/rhydian/self-supervised-project/model_weights/SSECEncoders'
    ENCODER_TYPE = VGGEncoder
    EMBEDDING_SIZE = 128
    USE_CUDA = True
    UPSAMPLE_FACTOR = 2 # amount feature maps scaled up for subpixel accuracy
    USE_CYCLIC = True # use cyclic point correlation
    USE_RANSAC = True
    RANSAC_TOLERANCE = 15
    USE_EQUAL_RES = True
    GRID_RES = 1
    NUM_DISCRIM_POINTS = int(300*GRID_RES**2) # number of correlating pairs to find
    ALLOW_SCALING=False


def set_up():
    global c
    train_ds = BothMidCoronalDataset(set_type='train', all_root=c.DATASET_ROOT, return_segmentations=False, equal_res=c.USE_EQUAL_RES, augment=c.TRAINING_AUGMENTATION)
    val_ds   = BothMidCoronalDataset(set_type='val'  , all_root=c.DATASET_ROOT, return_segmentations=False, equal_res=c.USE_EQUAL_RES, augment=False)
    test_ds  = BothMidCoronalDataset(set_type='test' , all_root=c.DATASET_ROOT, return_segmentations=False, equal_res=c.USE_EQUAL_RES, augment=False)
    dxa_model, mri_model, val_stats, epochs = load_models(c.ENCODER_TYPE, c.EMBEDDING_SIZE, True, c.MODEL_LOAD_PATH, c.USE_CUDA )
    return train_ds, val_ds, test_ds, dxa_model, mri_model, val_stats, epochs


def save_registration_examples(ds, dxa_model, mri_model):
    global c
    for idx in tqdm(range(10)):
        sample = ds[idx]
        tgt_img = sample['dxa_img']; tgt_model = dxa_model
        src_img = sample['mri_img']; src_model = mri_model
        src_inliers, tgt_inliers,M,_,_,_,error = get_matching_keypoints(src_img,tgt_img,src_model, tgt_model,
                                                          c.USE_CUDA, c.UPSAMPLE_FACTOR, c.USE_CYCLIC,c.NUM_DISCRIM_POINTS,
                                                          c.USE_RANSAC, c.RANSAC_TOLERANCE, c.GRID_RES, c.ALLOW_SCALING)




        fig, ax = show_matches(src_img, tgt_img,src_inliers, tgt_inliers)
        fig.tight_layout()
        fig.savefig(f'../images/example-keypoint-matches2/example_keypoint_match_{idx}.png')

        if c.USE_RANSAC:
            fig, axs = show_warped_perspective(src_img,tgt_img,M)
            fig.tight_layout()
            fig.savefig(f'../images/example-keypoint-matches2/alignment_match_{idx}.png')




c = config()
if __name__ == '__main__':
    train_ds, val_ds, test_ds, dxa_model, mri_model, val_stats, epochs = set_up()
    save_registration_examples(test_ds, dxa_model, mri_model)



