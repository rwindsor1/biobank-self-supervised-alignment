import sys, os, glob
import torch
import pandas as pd
from scipy.io import loadmat
from torch.utils.data import Dataset
import torchvision.transforms as t
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import augmentation_utils as AU

import pickle

class Both3DDataset(Dataset):
    def __init__(self, set_type='train',
                 all_root='/scratch/shared/beegfs/rhydian/UKBiobank/',
                 mri_root='Cleaned3DMRIs',
                 dxa_root='dxa',
                 single_dxa = False,
                 single_mri = False,
                 augment=False,
                 scan_lists_path='/users/rhydian/self-supervised-project/scan_lists'):

        super().__init__()
        assert set_type in ['train', 'val', 'test', 'all']
        if set_type == 'all':
            self.scans_df = pd.read_csv(os.path.join(scan_lists_path, f'biobank_mri_dxa_id_alignments.csv'))
        else:
            self.scans_df = pd.read_csv(os.path.join(scan_lists_path, f'biobank_mri_dxa_id_alignments_{set_type}.csv'))

        self.single_dxa = single_dxa
        self.single_mri = single_mri
        self.mri_root = os.path.join(all_root, mri_root, set_type)
        self.dxa_root = os.path.join(all_root, dxa_root)
        if set_type == 'all': raise NotImplementedError()
        self.augment = augment
        # self.blanking_augment = blanking_augment

    def __len__(self):
        return len(self.scans_df)

    def __getitem__(self, idx):
        try:
            scan_row = self.scans_df.iloc[idx]
            fat_mri_path = os.path.join(self.mri_root, scan_row['mri_filename'] + '_F.pkl')
            water_mri_path = os.path.join(self.mri_root, scan_row['mri_filename'] + '_W.pkl')
            bone_dxa_path = os.path.join(self.dxa_root,'bone', scan_row['dxa_filename'] + '.mat')
            tissue_dxa_path = os.path.join(self.dxa_root,'tissue', scan_row['dxa_filename'] + '.mat')
            # load dxas
            bone_dxa_img  = torch.Tensor(loadmat(bone_dxa_path)['scan']).float()
            if not self.single_dxa:
                tissue_dxa_img  = torch.Tensor(loadmat(tissue_dxa_path)['scan']).float()
            else:
                tissue_dxa_img = torch.zeros_like(bone_dxa_img)

            # load mris
            with open(fat_mri_path,'rb') as f:
                fat_mri_img  = torch.Tensor(pickle.load(f)['volume'])

            if not self.single_mri:
                with open(water_mri_path,'rb') as f:
                    water_mri_img  = torch.Tensor(pickle.load(f)['volume'])
            else:
                water_mri_img = torch.zeros_list(fat_mri_img)

        except Exception as E: 
            print(f'No luck, error {E}')
            return Both3DDataset.__getitem__(self, np.random.randint(self.scans_df.__len__()-1))

        
        output_dxa_shape = (1 + int(not(self.single_dxa)), 700, 276)
        output_mri_shape = (1 + int(not(self.single_mri)), 501, 156, 224)

        # parameters for augmentation
        ROT_LOW = 0
        ROT_HIGH = 0
        TRANS_LOW = -4
        TRANS_HIGH = 5
        ZOOM_LOW = 1 
        ZOOM_HIGH = 1
        CONTRAST_VAR = 0.2
        BRIGHTNESS_VAR = 0.2

        if self.augment:
            dxa_rot        = np.random.random()*(ROT_HIGH-ROT_LOW) + ROT_LOW
            mri_rot        = np.random.random()*(ROT_HIGH-ROT_LOW) + ROT_LOW
            dxa_delta_x    = np.random.randint(TRANS_LOW,TRANS_HIGH)
            dxa_delta_y    = np.random.randint(TRANS_LOW,TRANS_HIGH)
            mri_delta_x    = np.random.randint(TRANS_LOW,TRANS_HIGH)
            mri_delta_y    = np.random.randint(TRANS_LOW,TRANS_HIGH)
            mri_zoom       = np.random.random()*(ZOOM_HIGH - ZOOM_LOW) + ZOOM_LOW
            dxa_zoom       = np.random.random()*(ZOOM_HIGH - ZOOM_LOW) + ZOOM_LOW
            dxa_brightness = 1 + 2*BRIGHTNESS_VAR*(np.random.random()-0.5)
            mri_brightness = 1 + 2*BRIGHTNESS_VAR*(np.random.random()-0.5)
            dxa_contrast   = 1 + 2*CONTRAST_VAR*(np.random.random()-0.5)
            mri_contrast   = 1 + 2*CONTRAST_VAR*(np.random.random()-0.5)

        else:
            mri_rot = 0; dxa_rot = 0;
            dxa_delta_x = 0; dxa_delta_y = 0;
            mri_delta_x = 0; mri_delta_y = 0;
            dxa_zoom = 1; mri_zoom = 1;
            mri_brightness=1; dxa_brightness=1;
            dxa_contrast=1; mri_contrast=1

        if fat_mri_img.shape != water_mri_img.shape or tissue_dxa_img.shape != bone_dxa_img.shape:
            return Both3DDataset.__getitem__(self, np.random.randint(self.scans_df.__len__()-1))

        if not self.single_mri:
            mri_img = torch.stack([fat_mri_img, water_mri_img], dim=0)
        else:
            mri_img = fat_mri_img[None]
        if not self.single_dxa:
            dxa_img = torch.stack([bone_dxa_img, tissue_dxa_img], dim=0)

        if dxa_img.shape[1] != output_dxa_shape[1] or dxa_img.shape[2] != output_dxa_shape[2]:
            diff = (output_dxa_shape[1] - dxa_img.shape[1], output_dxa_shape[2] - dxa_img.shape[2])
            dxa_img = F.pad(dxa_img,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])

        if mri_img.shape[1] != output_mri_shape[1] or mri_img.shape[2] != output_mri_shape[2] or mri_img.shape[3] != output_mri_shape[3]:
            diff = (output_mri_shape[1] - mri_img.shape[1], output_mri_shape[2] - mri_img.shape[2], output_mri_shape[3] - mri_img.shape[3])
            mri_img = F.pad(mri_img,[int(np.floor(diff[2]/2)),int(np.ceil(diff[2]/2)),int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])

        mri_img  = AU.adjust_contrast(TF.adjust_brightness(mri_img, mri_brightness), mri_contrast)
        dxa_img  = AU.adjust_contrast(TF.adjust_brightness(dxa_img, dxa_brightness), 1)
        return {'dxa_img': dxa_img, 'mri_img': mri_img}

        
if __name__ == '__main__':
    ds = Both3DDataset()
    ds[0]

