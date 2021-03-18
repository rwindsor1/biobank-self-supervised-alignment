import sys, os, glob
sys.path.append('/users/rhydian/self-supervised-project')
import torch
import pandas as pd
from scipy.io import loadmat
from torch.utils.data import Dataset
import torchvision.transforms as t
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import augmentation_utils as AU
from gen_utils import *

class DXASegmentationDataset(Dataset):
    def __init__(self, set_type='train',
                 all_root='/scratch/shared/beegfs/rhydian/UKBiobank/',
                 dxa_root='dxa',
                 dxa_segmentation_root='dxa-segmentation3',
                 scan_lists_path='/users/rhydian/self-supervised-project/scan_lists',
                 augment=False, blanking_augment=False):
        super().__init__()
        assert set_type in ['train', 'val', 'test', 'all']
        if set_type == 'all':
            self.scans_df = pd.read_csv(os.path.join(scan_lists_path, f'biobank_mri_dxa_id_alignments.csv'))
        else:
            self.scans_df = pd.read_csv(os.path.join(scan_lists_path, f'biobank_mri_dxa_id_alignments_{set_type}.csv'))

        self.dxa_root = os.path.join(all_root, dxa_root)
        self.dxa_segmentation_root = os.path.join(all_root,dxa_segmentation_root)
        if set_type == 'all': raise NotImplementedError()
        self.augment = augment
        self.blanking_augment = blanking_augment


    def __getitem__(self, idx):
        try:
            scan_row = self.scans_df.iloc[idx]
            bone_dxa_path = os.path.join(self.dxa_root,'bone', scan_row['dxa_filename'] + '.mat')
            tissue_dxa_path = os.path.join(self.dxa_root,'tissue', scan_row['dxa_filename'] + '.mat')
            bone_dxa_img  = torch.Tensor(loadmat(bone_dxa_path)['scan']).float()
            tissue_dxa_img  = torch.Tensor(loadmat(tissue_dxa_path)['scan']).float()
            dxa_segmentations_path = os.path.join(self.dxa_segmentation_root,scan_row['dxa_filename']+'.npy')
            dxa_segmentations = torch.Tensor(np.load(dxa_segmentations_path)).permute(2,0,1)
        except Exception as E:
            print(E)
            return DXASegmentationDataset.__getitem__(self, np.random.randint(self.scans_df.__len__()-1))

        output_dxa_shape = (2,700,276)

        # parameters for augmentation
        ROT_LOW = -10
        ROT_HIGH = 10
        TRANS_LOW = -4
        TRANS_HIGH = 5
        ZOOM_LOW = 0.9
        ZOOM_HIGH = 1.1
        CONTRAST_VAR = 0.2
        BRIGHTNESS_VAR = 0.2

        if self.augment:
            dxa_rot        = np.random.random()*(ROT_HIGH-ROT_LOW) + ROT_LOW
            dxa_delta_x    = np.random.randint(TRANS_LOW,TRANS_HIGH)
            dxa_delta_y    = np.random.randint(TRANS_LOW,TRANS_HIGH)
            dxa_zoom       = np.random.random()*(ZOOM_HIGH - ZOOM_LOW) + ZOOM_LOW
            dxa_brightness = 1 + 2*BRIGHTNESS_VAR*(np.random.random()-0.5)
            dxa_contrast   = 1 + 2*CONTRAST_VAR*(np.random.random()-0.5)

        else:
            dxa_rot = 0;
            dxa_delta_x = 0; dxa_delta_y = 0;
            mri_delta_x = 0; mri_delta_y = 0;
            dxa_zoom = 1;
            dxa_brightness=1;
            dxa_contrast=1;

        if tissue_dxa_img.shape != bone_dxa_img.shape:
            return BothMidCoronalDataset.__getitem__(self, np.random.randint(self.scans_df.__len__()-1))

        dxa_img = torch.stack([bone_dxa_img, tissue_dxa_img], dim=0)
        assert dxa_img.shape[1:] == dxa_segmentations.shape[1:], f"Segmentations not the same size: {dxa_img.shape[1:]}, {dxa_segmentations.shape[1:]}"


        dxa_img = TF.rotate(dxa_img[None], dxa_rot)[0]
        dxa_segmentations = TF.rotate(dxa_segmentations[None], dxa_rot)[0]

        # crop to correct size
        if dxa_img.shape[1] != output_dxa_shape[1] or dxa_img.shape[2] != output_dxa_shape[2]:
            diff = (output_dxa_shape[1] - dxa_img.shape[1], output_dxa_shape[2] - dxa_img.shape[2])
            dxa_img = F.pad(dxa_img,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])
            dxa_segmentations = F.pad(dxa_segmentations,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])

        dxa_img  = AU.adjust_contrast(TF.adjust_brightness(dxa_img, dxa_brightness), 1)

        return {'dxa_img': dxa_img, 'dxa_segmentation': dxa_segmentations}

    def __len__(self):
        return len(self.scans_df)


if __name__ == '__main__':
    ds = DXASegmentationDataset(augment=True)
    for idx in ds:
        sample = ds[idx]
        plt.imshow(grayscale(sample['dxa_img'][0])+red(sample['dxa_segmentations'][1])+blue(sample['dxa_segmentations'][0]))
        plt.show()

        import pdb;pdb.set_trace()

