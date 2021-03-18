import sys,os,glob
import pickle as pkl
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.append('/users/rhydian/self-supervised-project')

from datasets.BothMidCoronalDataset import BothMidCoronalDataset
from models.SSECEncoders import VGGEncoder
# configuration for alignment
class config():
    TRAINING_AUGMENTATION=False
    DATASET_ROOT = '/scratch/shared/beegfs/rhydian/UKBiobank'
    OUTPUT_PATH = os.path.join(DATASET_ROOT, 'files-for-annotation')
    TRAINING_AUGMENTATION = False
    NUM_SCANS=100


def set_up():
    global c
    train_ds = BothMidCoronalDataset(set_type='train', all_root=c.DATASET_ROOT, return_segmentations=False, augment=c.TRAINING_AUGMENTATION)
    val_ds   = BothMidCoronalDataset(set_type='val'  , all_root=c.DATASET_ROOT, return_segmentations=False, augment=False)
    test_ds  = BothMidCoronalDataset(set_type='test' , all_root=c.DATASET_ROOT, return_segmentations=False, augment=False)
    return train_ds, val_ds, test_ds

c = config()
if __name__ == '__main__':
    train_ds, val_ds, test_ds = set_up()

    for idx in tqdm(range(c.NUM_SCANS)):
        sample = test_ds[idx]
        subject_info = test_ds.scans_df.iloc[idx]
        dxa_id = subject_info['dxa_filename']
        mri_id = subject_info['mri_filename']
        subject_identifier = '_'.join([subject_info['id'], dxa_id, mri_id])
        dxa_slice = sample['dxa_img'][0]
        mri_slice = sample['mri_img'][0]
        mri_path = os.path.join(c.DATASET_ROOT, 'stitched_mris', mri_id +'_F.pkl')
        with open(mri_path,'rb') as f:
            mri_vol = pkl.load(f)['volume']
        affine_mat = nib.orientations.inv_ornt_aff(nib.orientations.axcodes2ornt(('S','A','L')),mri_vol.shape)
        if idx == 0: print(affine_mat)
        mri_nii = nib.Nifti1Image(mri_vol, affine=affine_mat)
        nib.save(mri_nii,os.path.join(c.OUTPUT_PATH,'mri-niis', subject_identifier+'.nii.gz'))
        dxa_img = Image.fromarray((dxa_slice.numpy()*255)).convert('L')
        mri_img = Image.fromarray((mri_slice.numpy()*255)).convert('L')
        mri_img.save(os.path.join(c.OUTPUT_PATH, 'mri-slices', subject_identifier + '.png'))
        dxa_img.save(os.path.join(c.OUTPUT_PATH, 'dxa-slices', subject_identifier + '.png'))
        assert (mri_vol.shape[0], mri_vol.shape[2]) == mri_slice.shape, f'Different shape MRIs {mri_vol.shape}, {mri_slice.shape}'


        #import pdb; pdb.set_trace()
