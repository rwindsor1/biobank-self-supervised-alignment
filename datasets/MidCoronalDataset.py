import sys, os, glob
import torch
import pandas as pd
from scipy.io import loadmat
from torch.utils.data import Dataset
import torchvision.transforms as t
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

class MidCoronalDataset(Dataset):
    def __init__(self, set_type='train',
                 all_root='/scratch/shared/beegfs/rhydian/UKBiobank/',
                 mri_root='SynthesisedMRISlices2',
                 dxa_root='dxa/bone',
                 scan_lists_path='/users/rhydian/self-supervised-project/scan_lists',
                 augment=False, blanking_augment=False):
        super(MidCoronalDataset, self).__init__()
        assert set_type in ['train', 'val', 'test', 'all']
        if set_type == 'all':
            self.scans_df = pd.read_csv(os.path.join(scan_lists_path, f'biobank_mri_dxa_id_alignments.csv'))
        else:
            self.scans_df = pd.read_csv(os.path.join(scan_lists_path, f'biobank_mri_dxa_id_alignments_{set_type}.csv'))

        self.mri_root = os.path.join(all_root, mri_root, set_type)
        self.dxa_root = os.path.join(all_root, dxa_root)
        if set_type == 'all': raise NotImplementedError()
        self.augment = augment
        self.blanking_augment = blanking_augment


    def __getitem__(self, idx):
        try:
            scan_row = self.scans_df.iloc[idx]
            mri_path = os.path.join(self.mri_root, scan_row['mri_filename'] + '_F')
            dxa_path = os.path.join(self.dxa_root, scan_row['dxa_filename'] + '.mat')
            dxa_img  = torch.Tensor(loadmat(dxa_path)['scan']).float()
            mri_img  = torch.load(mri_path)
        except: 
            return MidCoronalDataset.__getitem__(self, np.random.randint(self.scans_df.__len__()-1))

        output_dxa_shape = (700,276)
        output_mri_shape = (501,224)

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


        dxa_img = TF.rotate(dxa_img[None,None], dxa_rot)[0,0]
        mri_img = TF.rotate(mri_img[None,None], mri_rot)[0,0]


        # crop to correct size
        if dxa_img.shape[0] != output_dxa_shape[0] or dxa_img.shape[1] != output_dxa_shape[1]:
            diff = (output_dxa_shape[0] - dxa_img.shape[0], output_dxa_shape[1] - dxa_img.shape[1])
            dxa_img = F.pad(dxa_img,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])

        if mri_img.shape[0] != output_mri_shape[0] or mri_img.shape[1] != output_mri_shape[1]:
            diff = (output_mri_shape[0] - mri_img.shape[0], output_mri_shape[1] - mri_img.shape[1])
            mri_img = F.pad(mri_img,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])

        mri_img  = TF.adjust_contrast(TF.adjust_brightness(torch.cat([mri_img[None,None]]*3, axis=1), mri_brightness), mri_contrast)[0,0][None]
        dxa_img  = TF.adjust_contrast(TF.adjust_brightness(torch.cat([dxa_img[None,None]]*3, axis=1), dxa_brightness), 1)[0,0][None]

        if mri_zoom <= 1:
            mri_img = TF.resized_crop(mri_img, int(np.random.random()*(1-mri_zoom)*mri_img.shape[-2]),
                                    int(np.random.random()*(1-mri_zoom)*mri_img.shape[-1]),
                                    int(mri_zoom*mri_img.shape[-2]), int(mri_zoom*mri_img.shape[-1]),
                                    output_mri_shape)
        else:
            added_height = int(np.floor((mri_zoom - 1)*mri_img.shape[1]))
            added_width  = (mri_zoom - 1)*mri_img.shape[2]
            rand_h_1     = int(np.floor(added_height*np.random.random()))
            rand_w_1     = int(np.floor(added_width*np.random.random()))
            rand_h_2     = int(np.ceil(added_height - rand_h_1))
            rand_w_2     = int(np.ceil(added_width - rand_w_1))
            mri_img      = F.pad(mri_img, [rand_w_1, rand_w_2, rand_h_1, rand_h_2]) 
            mri_img      = F.interpolate(mri_img[None], output_mri_shape)[0]

        if dxa_zoom <= 1:
            dxa_img = TF.resized_crop(dxa_img, int(np.random.random()*(1-dxa_zoom)*dxa_img.shape[-2]),
                                    int(np.random.random()*(1-dxa_zoom)*dxa_img.shape[-1]),
                                    int(dxa_zoom*dxa_img.shape[-2]), int(dxa_zoom*dxa_img.shape[-1]),
                                    output_dxa_shape)
        else:
            added_height = int(np.floor((dxa_zoom -1)*dxa_img.shape[1]))
            added_width  = (dxa_zoom -1)*dxa_img.shape[2]
            rand_h_1     = int(np.floor(added_height*np.random.random()))
            rand_w_1     = int(np.floor(added_width*np.random.random()))
            rand_h_2     = int(np.ceil(added_height - rand_h_1))
            rand_w_2     = int(np.ceil(added_width - rand_w_1))
            dxa_img      = F.pad(dxa_img, [rand_w_1, rand_w_2, rand_h_1, rand_h_2]) 
            dxa_img      = F.interpolate(dxa_img[None], output_dxa_shape)[0]

        if self.blanking_augment:
                if np.random.random() > 0.25:
                    rand_h = np.random.randint(np.ceil(dxa_img.shape[1]/5),np.ceil(dxa_img.shape[1]/4))
                    rand_x = np.random.randint(0, dxa_img.shape[1] - rand_h)
                    dxa_img[:,rand_x:rand_x+rand_h] = 0

                if np.random.random() > 0.25:
                    rand_h = np.random.randint(np.ceil(mri_img.shape[1]/5),np.ceil(mri_img.shape[1]/4))
                    rand_x = np.random.randint(0, mri_img.shape[1] - rand_h)
                    mri_img[:,rand_x:rand_x+rand_h] = 0


        return {'dxa_img': dxa_img, 'mri_img': mri_img}

    def __len__(self):
        return len(self.scans_df)

class MidCoronalHardNegativesDataset(MidCoronalDataset):
    def __init__(self, hard_negative_list: list, set_type='train',
                 all_root='/scratch/shared/beegfs/rhydian/UKBiobank/',
                 mri_root='SynthesisedMRISlices2',
                 dxa_root='dxa/bone',
                 scan_lists_path='/users/rhydian/self-supervised-project/scan_lists',
                 augment=False, blanking_augment=False):
        super(MidCoronalHardNegativesDataset, self).__init__(set_type, all_root, mri_root, dxa_root, 
                                               scan_lists_path, augment, blanking_augment)
        self.hard_negative_list = hard_negative_list

    def __getitem__(self, idx):
        hard_negative_idxs = self.hard_negative_list[idx]
        dxa_idx, mri_idx = hard_negative_idxs
        sample = super().__getitem__(dxa_idx)
        dxa_img1 = sample['dxa_img']
        mri_img1 = sample['mri_img']
        other_sample = super().__getitem__(mri_idx)
        dxa_img2 = other_sample['dxa_img']
        mri_img2 = other_sample['mri_img']
        return {'dxa_img1': dxa_img1, 'mri_img1': mri_img1, 
                'dxa_img2': dxa_img2, 'mri_img2': mri_img2}

    def __len__(self):
        return len(self.hard_negative_list)
        
        

if __name__ == '__main__':
    import pickle

    import matplotlib.pyplot as plt
    # ds = MidCoronalDataset(augment=True, blanking_augment = False)
    unaugmented_ds = MidCoronalDataset(augment=False, blanking_augment = False)
    for idx in range(len(unaugmented_ds)):
        unaugment_sample = unaugmented_ds[idx]
        plt.figure(figsize=(10,10))
        dxa_img = unaugment_sample['dxa_img']
        mri_img = unaugment_sample['mri_img']
        plt.title('MRI')
        plt.imshow(mri_img[0], cmap='gray')
        plt.show()
        # unaugment_dxa_img = unaugment_sample['dxa_img']
        # unaugment_mri_img = unaugment_sample['mri_img']
        # plt.subplot(221)
        # plt.title('Augment DXA')
        # plt.imshow(dxa_img[0], cmap='gray')
        # plt.subplot(222)
        # plt.title('Augment MRI')
        # plt.imshow(mri_img[0], cmap='gray')
        # plt.subplot(223)
        # plt.title('No Augment DXA')
        # plt.imshow(unaugment_dxa_img[0], cmap='gray')
        # plt.subplot(224)
        # plt.title('No Augment MRI')
        # plt.imshow(unaugment_mri_img[0], cmap='gray')
        # plt.show()

