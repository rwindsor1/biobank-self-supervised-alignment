import sys
import os
import glob
from os.path import dirname,abspath,join, basename
from numpy.lib.npyio import savez_compressed
import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as t
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def get_seqs(scan_obj, seq_names):
    out_img = []
    if isinstance(seq_names,str): seq_names = list(seq_names)
    for seq_name in seq_names:
        out_img.append(scan_obj[seq_name])
    return torch.cat(out_img,dim=1)

def resample_scans(scan_obj, resolution=2,transpose=False):
    scaling_factors = resolution/np.array(scan_obj['pixel_spacing'])
    for seq_name in scan_obj.keys():
        if seq_name == 'pixel_spacing': scan_obj['pixel_spacing'] = [resolution]*2 
        else:
            scan  = torch.Tensor(scan_obj[seq_name])[None,None]
            scan_obj[seq_name] = F.interpolate(scan,
                                               scale_factor=list(scaling_factors),
                                               recompute_scale_factor=False,
                                               mode='bicubic')
            if transpose:
                scan_obj[seq_name] = scan_obj[seq_name].permute(0,1,3,2)
    return scan_obj

def pad_to_size(scan_img : torch.Tensor, output_shape : tuple):
    ''' Pads or crops image to a given size'''
    if (scan_img.shape[1] != output_shape[1]) or (scan_img[2] != output_shape[2]):
        diff = (output_shape[1] - scan_img.shape[1], output_shape[2] - scan_img.shape[2])
        scan_img = F.pad(scan_img,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])
    return scan_img

def normalise_channels(scan_img : torch.Tensor, eps : float = 1e-5):
    scan_min = scan_img.flatten(start_dim=-2).min(dim=-1)[0][:,None,None]
    scan_max = scan_img.flatten(start_dim=-2).max(dim=-1)[0][:,None,None]
    return (scan_img-scan_min)/(scan_max-scan_min + eps)

class CoronalScanPairsDataset(Dataset):
    def __init__(self,
                 root : str,
                 set_type : str,
                 augment : bool = False,
                 mri_seqs : list = ['fat_scan','water_scan'],
                 dxa_seqs : list = ['bone','tissue'],
                 mri_dirname : str = 'mri-mid-corr-slices',
                 dxa_dirname : str = 'dxas-processed',
                 reload_if_fail : bool = False):
        super().__init__()
        assert set_type in ['train', 'val', 'test', 'all']
        self.set_type = set_type

        self.mri_root = join(root, mri_dirname)
        self.dxa_root = join(root, dxa_dirname)
        self.mri_seqs = mri_seqs
        self.dxa_seqs = dxa_seqs

        patients = []

        all_mris = os.listdir(self.mri_root)
        all_dxas = os.listdir(self.dxa_root)
        mri_subjects = [basename(x).split('_')[0] for x in all_mris]
        dxa_subjects = [basename(x).split('_')[0] for x in all_dxas]
        intersection_subjects = set(mri_subjects).intersection(dxa_subjects)
        # load in the dataset split file
        if set_type is not 'all':
            split_file = join(abspath(__file__+'/../../'),
                                      'assets',
                                      f'{self.set_type}_subjects.txt')
            split_subjects = [x.replace('\n','') for x in open(split_file,'r')]
            intersection_subjects = intersection_subjects.intersection(split_subjects)
        intersection_subjects = sorted(list(intersection_subjects))

        # now we have subjects, construct scan pairs
        self.pairs = []
        for subject in intersection_subjects:
            subject_dxa_scans = [x for x in all_dxas if subject in x]
            subject_mri_scans = [x for x in all_mris if subject in x]
            self.pairs.append([subject_dxa_scans[0],subject_mri_scans[0]])

        self.augment = augment

    def __len__(self):
        return len(self.scans_df)

    def __getitem__(self, idx):
        dxa_path, mri_filename = self.pairs[idx]
        dxa_fp = glob.glob(os.path.join(self.dxa_root,dxa_path,'*.pkl'))[0]
        mri_fp = os.path.join(self.mri_root, mri_filename)

        with open(dxa_fp,'rb') as f:
            dxa_scan = pickle.load(f)
        with open(mri_fp,'rb') as f:
            mri_scan =pickle.load(f)

        if self.augment:
            mri_mid_slice = np.random.randint(mri_scan['fat_scan'].shape[1]) 
        else:
            mri_mid_slice = mri_scan['fat_scan'].shape[1]//2

        # get mid sag from mri scan
        for sequence in ['fat_scan','water_scan']:
            mri_scan[sequence] = mri_scan[sequence][:,mri_mid_slice]
        mri_scan['pixel_spacing'] = np.array(mri_scan['pixel_spacing'])[[0,2]]

        # resample both scans to 2x2 mm
        dxa_scan['pixel_spacing'] = [2,2]
        mri_scan=resample_scans(mri_scan,transpose=True)
        dxa_scan=resample_scans(dxa_scan,transpose=False)


        # parameters for augmentation
        ROT_LOW = -10
        ROT_HIGH = 10
        TRANS_LOW = -4
        TRANS_HIGH = 5
        # no zoom
        CONTRAST_VAR = 0.2
        BRIGHTNESS_VAR = 0.2

        if self.augment:
            dxa_rot        = np.random.random()*(ROT_HIGH-ROT_LOW) + ROT_LOW
            mri_rot        = np.random.random()*(ROT_HIGH-ROT_LOW) + ROT_LOW
            dxa_delta_x    = np.random.randint(TRANS_LOW,TRANS_HIGH)
            dxa_delta_y    = np.random.randint(TRANS_LOW,TRANS_HIGH)
            mri_delta_x    = np.random.randint(TRANS_LOW,TRANS_HIGH)
            mri_delta_y    = np.random.randint(TRANS_LOW,TRANS_HIGH)
            dxa_brightness = 1 + 2*BRIGHTNESS_VAR*(np.random.random()-0.5)
            mri_brightness = 1 + 2*BRIGHTNESS_VAR*(np.random.random()-0.5)
            dxa_gamma   = random.choice([1,1,0.5,1.5])
            mri_gamma   = random.choice([1,1,0.5,1.5])

        else:
            mri_rot = 0; dxa_rot = 0;
            dxa_delta_x = 0; dxa_delta_y = 0;
            mri_delta_x = 0; mri_delta_y = 0;
            mri_brightness=1; dxa_brightness=1;
            mri_gamma=1; dxa_gamma=1


        mri_img = get_seqs(mri_scan, self.mri_seqs)
        dxa_img = get_seqs(dxa_scan, self.dxa_seqs)

        # augment 
        dxa_img = TF.affine(dxa_img, dxa_rot,(dxa_delta_x,dxa_delta_y),1,(0,0))[0]
        mri_img = TF.affine(mri_img, mri_rot,(mri_delta_x,mri_delta_y),1,(0,0))[0]

        # normalise scans
        dxa_img = normalise_channels(dxa_img)
        mri_img = normalise_channels(mri_img)

        # adjust brightness/contrast
        mri_img  = TF.adjust_gamma(TF.adjust_brightness(mri_img, mri_brightness), mri_gamma, gain=1)
        dxa_img  = TF.adjust_gamma(TF.adjust_brightness(dxa_img, dxa_brightness), dxa_gamma, gain=1)

        output_dxa_shape = (2,1000,300)
        output_mri_shape = (2,501,224)

        # crop to correct size
        mri_img = pad_to_size(mri_img, output_mri_shape)
        dxa_img = pad_to_size(dxa_img, output_dxa_shape)



        return_dict = {'dxa_img': dxa_img, 'mri_img': mri_img,
                       'mri_filename': mri_fp, 'dxa_filename': dxa_fp}

        return return_dict
