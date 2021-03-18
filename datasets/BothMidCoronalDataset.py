import sys, os, glob
import torch
import pandas as pd
from scipy.io import loadmat
from torch.utils.data import Dataset
import torchvision.transforms as t
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
sys.path.append('/users/rhydian/self-supervised-project')
import augmentation_utils as AU

class BothMidCoronalDataset(Dataset):
    def __init__(self, set_type='train',
                 all_root='/scratch/shared/beegfs/rhydian/UKBiobank/',
                 mri_root='SynthesisedMRISlices2',
                 dxa_root='dxa',
                 dxa_segmentation_root='dxa-segmentation3',
                 scan_lists_path='/users/rhydian/self-supervised-project/scan-lists',
                 augment=False, blanking_augment=False, return_segmentations=True,
                 single_dxa=False, single_mri=False,
                 crop_scans=True, reload_if_fail=True, equal_res=True,return_scan_info=False,
                 relative_scan_alignments=None):
        super().__init__()
        assert set_type in ['train', 'val', 'test', 'all']
        if relative_scan_alignments is not None:
            assert equal_res == True
        if set_type == 'all':
            self.scans_df = pd.read_csv(os.path.join(scan_lists_path, f'biobank_mri_dxa_id_alignments.csv'))
        else:
            self.scans_df = pd.read_csv(os.path.join(scan_lists_path, f'biobank_mri_dxa_id_alignments_{set_type}.csv'))

        self.mri_root = os.path.join(all_root, mri_root, set_type)
        self.dxa_root = os.path.join(all_root, dxa_root)
        self.dxa_segment_root = os.path.join(all_root, dxa_segmentation_root)
        if set_type == 'all': raise NotImplementedError()
        self.augment = augment
        self.blanking_augment = blanking_augment
        self.return_segmentation = return_segmentations
        self.crop_scans = crop_scans
        self.reload_if_fail = reload_if_fail
        self.equal_res = equal_res
        self.return_scan_info = return_scan_info
        self.single_dxa=single_dxa
        self.single_mri=single_mri
        if relative_scan_alignments is not None:
            self.relative_scan_alignments = pd.read_csv(relative_scan_alignments, names=['mri_filename','dxa_filename','angle','t_x','t_y'])
        else:
            self.relative_scan_alignments = None


    def __getitem__(self, idx):
        try:
            scan_row = self.scans_df.iloc[idx]
            rescale_factor = 1/0.911
            fat_mri_path = os.path.join(self.mri_root, scan_row['mri_filename'] + '_F')
            water_mri_path = os.path.join(self.mri_root, scan_row['mri_filename'] + '_W')
            bone_dxa_path = os.path.join(self.dxa_root,'bone', scan_row['dxa_filename'] + '.mat')
            tissue_dxa_path = os.path.join(self.dxa_root,'tissue', scan_row['dxa_filename'] + '.mat')
            segmentation_path = os.path.join(self.dxa_segment_root, scan_row['dxa_filename'] + '.npy')
            bone_dxa_img  = torch.Tensor(loadmat(bone_dxa_path)['scan']).float()
            tissue_dxa_img  = torch.Tensor(loadmat(tissue_dxa_path)['scan']).float()
            if self.equal_res:
                tissue_dxa_img = F.interpolate(tissue_dxa_img[None,None],scale_factor=rescale_factor, recompute_scale_factor=False)[0,0]
                bone_dxa_img =   F.interpolate(bone_dxa_img[None,None],  scale_factor=rescale_factor, recompute_scale_factor=False)[0,0]
            if self.return_segmentation:
                segmentation_img = torch.Tensor(np.load(segmentation_path)).permute(2,0,1)
                if self.equal_res:
                    segmentation_img = F.interpolate(segmentation_img[None], scale_factor=rescale_factor, recompute_scale_factor=False)[0]
            fat_mri_img  = torch.load(fat_mri_path)
            water_mri_img  = torch.load(water_mri_path)

        except Exception as e:
            if self.reload_if_fail:
                return BothMidCoronalDataset.__getitem__(self, np.random.randint(self.scans_df.__len__()-1))
            else: print(e)
            exit()

        if self.equal_res:
            output_dxa_shape = (2,800,300)
        else:
            output_dxa_shape = (2,700,276)
        output_mri_shape = (2,501,224)

        # parameters for augmentation
        ROT_LOW = -0
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
            return BothMidCoronalDataset.__getitem__(self, np.random.randint(self.scans_df.__len__()-1))

        if self.single_mri==1:
            mri_img=fat_mri_img[None]
        elif self.single_mri==2:
            mri_img=water_mri_img[None]
        else:
            mri_img = torch.stack([fat_mri_img, water_mri_img], dim=0)

        if self.single_dxa==1:
            dxa_img = bone_dxa_img[None]
        elif self.single_dxa==2:
            dxa_img = tissue_dxa_img[None]
        else:
            dxa_img = torch.stack([bone_dxa_img, tissue_dxa_img], dim=0)

        dxa_img = TF.rotate(dxa_img[None], dxa_rot)[0]
        mri_img = TF.rotate(mri_img[None], mri_rot)[0]
        if self.return_segmentation:
            segmentation_img = TF.rotate(segmentation_img[None],dxa_rot)[0]



        # crop to correct size
        if (dxa_img.shape[1] != output_dxa_shape[1] or dxa_img.shape[2] != output_dxa_shape[2]) and self.crop_scans:
            diff = (output_dxa_shape[1] - dxa_img.shape[1], output_dxa_shape[2] - dxa_img.shape[2])
            dxa_img = F.pad(dxa_img,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])
            if self.return_segmentation:
                segmentation_img = F.pad(segmentation_img,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])

        if (mri_img.shape[1] != output_mri_shape[1] or mri_img.shape[2] != output_mri_shape[2]) and self.crop_scans:
            diff = (output_mri_shape[1] - mri_img.shape[1], output_mri_shape[2] - mri_img.shape[2])
            mri_img = F.pad(mri_img,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])

        mri_img  = AU.adjust_contrast(TF.adjust_brightness(mri_img, mri_brightness), mri_contrast)
        dxa_img  = AU.adjust_contrast(TF.adjust_brightness(dxa_img, dxa_brightness), 1)

        if self.crop_scans:
            if mri_zoom < 1:
                mri_img = TF.resized_crop(mri_img, int(np.random.random()*(1-mri_zoom)*mri_img.shape[-2]),
                                        int(np.random.random()*(1-mri_zoom)*mri_img.shape[-1]),
                                        int(mri_zoom*mri_img.shape[-2]), int(mri_zoom*mri_img.shape[-1]),
                                        output_mri_shape[1:])
            else:
                added_height = int(np.floor((mri_zoom - 1)*mri_img.shape[1]))
                added_width  = (mri_zoom - 1)*mri_img.shape[2]
                rand_h_1     = int(np.floor(added_height*np.random.random()))
                rand_w_1     = int(np.floor(added_width*np.random.random()))
                rand_h_2     = int(np.ceil(added_height - rand_h_1))
                rand_w_2     = int(np.ceil(added_width - rand_w_1))
                mri_img      = F.pad(mri_img, [rand_w_1, rand_w_2, rand_h_1, rand_h_2]) 
                mri_img      = F.interpolate(mri_img[None], output_mri_shape[1:])[0]
            if dxa_zoom <= 1:
                dxa_img = TF.resized_crop(dxa_img, int(np.random.random()*(1-dxa_zoom)*dxa_img.shape[-2]),
                                        int(np.random.random()*(1-dxa_zoom)*dxa_img.shape[-1]),
                                        int(dxa_zoom*dxa_img.shape[-2]), int(dxa_zoom*dxa_img.shape[-1]),
                                        output_dxa_shape[1:])
            else:
                added_height = int(np.floor((dxa_zoom -1)*dxa_img.shape[1]))
                added_width  = (dxa_zoom -1)*dxa_img.shape[2]
                rand_h_1     = int(np.floor(added_height*np.random.random()))
                rand_w_1     = int(np.floor(added_width*np.random.random()))
                rand_h_2     = int(np.ceil(added_height - rand_h_1))
                rand_w_2     = int(np.ceil(added_width - rand_w_1))
                dxa_img      = F.pad(dxa_img, [rand_w_1, rand_w_2, rand_h_1, rand_h_2]) 
                dxa_img      = F.interpolate(dxa_img[None], output_dxa_shape[1:])[0]

            if self.blanking_augment:
                    if np.random.random() > 0.25:
                        rand_h = np.random.randint(np.ceil(dxa_img.shape[1]/5),np.ceil(dxa_img.shape[1]/4))
                        rand_x = np.random.randint(0, dxa_img.shape[1] - rand_h)
                        dxa_img[:,rand_x:rand_x+rand_h] = 0

                    if np.random.random() > 0.25:
                        rand_h = np.random.randint(np.ceil(mri_img.shape[1]/5),np.ceil(mri_img.shape[1]/4))
                        rand_x = np.random.randint(0, mri_img.shape[1] - rand_h)
                        mri_img[:,rand_x:rand_x+rand_h] = 0


        return_dict = {'dxa_img': dxa_img, 'mri_img': mri_img}
        if self.return_segmentation:
            if segmentation_img.shape[-2]==800 and segmentation_img.shape[-1]==300:
                return_dict['target'] = segmentation_img
            else:
                return BothMidCoronalDataset.__getitem__(self, np.random.randint(self.scans_df.__len__()-1))


        if self.return_scan_info:
            return_dict['mri_filename'] = scan_row['mri_filename']
            return_dict['dxa_filename'] = scan_row['dxa_filename']

        if self.relative_scan_alignments is not None:
            self.relative_scan_alignments['mri_filename']
            row = self.relative_scan_alignments[self.relative_scan_alignments['mri_filename']==scan_row['mri_filename']]
            if len(row) != 1:
                return BothMidCoronalDataset.__getitem__(self, np.random.randint(self.scans_df.__len__()-1))
            else:
                angle = row['angle'].item()
                t_x = row['t_x'].item()
                t_y = row['t_y'].item()
                return_dict['angle'] = angle; return_dict['t_x']=t_x; return_dict['t_y']=t_y

        return return_dict

    def __len__(self):
        return len(self.scans_df)

class BothMidCoronalHardNegativesDataset(BothMidCoronalDataset):
    def __init__(self, hard_negative_list: list, set_type='train',
                 all_root='/scratch/shared/beegfs/rhydian/UKBiobank/',
                 mri_root='SynthesisedMRISlices2',
                 dxa_root='dxa',
                 scan_lists_path='/users/rhydian/self-supervised-project/scan-lists',
                 augment=False, blanking_augment=False):
        super(BothMidCoronalHardNegativesDataset, self).__init__(set_type, all_root, mri_root, dxa_root, 
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

class BothMidCoronalRegistrationDataset(BothMidCoronalDataset):
    def __init__(self, set_type='train',
                 all_root='/scratch/shared/beegfs/rhydian/UKBiobank/',
                 mri_root='SynthesisedMRISlices2',
                 dxa_root='dxa',
                 dxa_segmentation_root='dxa-segmentation3',
                 scan_lists_path='/users/rhydian/self-supervised-project/scan-lists',
                 relative_scan_alignments='/users/rhydian/self-supervised-project/scan-registration/scan_relative_transforms.csv',
                 rand_angle=5,
                 rand_translation=20,
                 ):
        super().__init__(set_type,
                               all_root,
                               mri_root,
                               dxa_root,
                               dxa_segmentation_root, 
                               scan_lists_path,
                               equal_res=True,
                               relative_scan_alignments=relative_scan_alignments)
        self.rand_angle =rand_angle
        self.rand_translation = rand_translation
        
    def rt_and_t(self,img, angle, t_x, t_y):
        return TF.affine(TF.rotate(img[None],angle,center=(0,0)),0,(t_y,t_x),1,(0,0))[0]

    def t_and_rt(self, img, angle, t_x, t_y):
        return TF.rotate(TF.affine(img[None],0,(t_y,t_x),1,(0,0)),angle,center=(0,0))[0]
    def __len__(self):
        return super().__len__()

    def __getitem__(self,idx):
        sample = super().__getitem__(idx)
        angle_orig = sample['angle']
        t_x_orig = sample['t_x']
        t_y_orig = sample['t_y']

        # glob_rotation = (np.random.random()-0.5)*self.glob_angle
        # glob_t_x = (np.random.random()-0.5)*self.glob_translation
        # glob_t_y = (np.random.random()-0.5)*self.glob_translation
        # dxa_img1 = self.rt_and_t(sample['dxa_img'],glob_rotation,glob_t_x, glob_t_y)
        # mri_img1 = self.rt_and_t(sample['mri_img'],glob_rotation,glob_t_x, glob_t_y)

        dxa_img1 = sample['dxa_img']
        mri_img1 = sample['mri_img']

        angle_aug = (np.random.random()-0.5)*self.rand_angle
        t_x_aug    = (np.random.random()-0.5)*self.rand_translation
        t_y_aug    = (np.random.random()-0.5)*self.rand_translation
        dxa_img2  = self.rt_and_t(dxa_img1,angle_aug, t_x_aug, t_y_aug)

        return {'dxa_img1':dxa_img1,'mri_img1':mri_img1,'dxa_img2':dxa_img2,
                'angle_orig':angle_orig,'t_x_orig': sample['t_x'],  
                't_y_orig':t_y_orig,
                'angle_aug': -angle_aug, 't_x_aug': -t_x_aug, 't_y_aug':-t_y_aug}        

def combine_rotations(a_1,t_x1, t_y1, a_2, t_x2, t_y2):
    # a1 is the first applied rot
    r_2= a_2*np.pi/180
    return [a_1+a_2, t_x2+t_x1*np.cos(r_2)-t_y1*np.sin(r_2), t_y2 + t_y1*np.cos(r_2) + t_x1*np.sin(r_2)]

if __name__ == '__main__':
    import pickle
    from torchvision.utils import make_grid, save_image
    from gen_utils import *

    import matplotlib.pyplot as plt
    # ds = MidCoronalDataset(augment=True, blanking_augment = False)
    ds = BothMidCoronalRegistrationDataset()
    for sample in ds:
        d_o = sample['dxa_img1']
        new_img1 = ds.rt_and_t(sample['dxa_img2'],-sample['angle_aug'],-sample['t_x_aug'], -sample['t_y_aug'])
        new_img2 = ds.t_and_rt(sample['dxa_img2'],-sample['angle_aug'],-sample['t_x_aug'], -sample['t_y_aug'])
        plt.subplot(141)
        plt.imshow(grayscale(sample['dxa_img1'][0]))
        plt.subplot(142)
        warped_dxa = ds.rt_and_t(sample['dxa_img1'],sample['angle_orig'],sample['t_x_orig'],sample['t_y_orig'])
        plt.imshow(red(warped_dxa[0,:501,:224])+grayscale(sample['mri_img1'][0]))
        plt.subplot(143)
        new_ang, new_t_x, new_t_y = combine_rotations(sample['angle_aug'],sample['t_x_aug'],sample['t_y_aug'],sample['angle_orig'],sample['t_x_orig'],sample['t_y_orig'])
        warped_dxa2 = ds.rt_and_t(sample['dxa_img2'],new_ang, new_t_x, new_t_y)
        plt.imshow(red(warped_dxa2[0,:501,:224])+grayscale(sample['mri_img1'][0]))
        plt.subplot(144)
        plt.imshow((warped_dxa-warped_dxa2)[0])
        #plt.imshow(grayscale(new_img2[0]-d_o[0]))
        plt.savefig('test.png')
        os.system('imgcat test.png')


    # for idx in range(len(unaugmented_ds)):
    #     unaugment_sample = unaugmented_ds[idx]
    #     plt.figure(figsize=(10,10))
    #     dxa_img = unaugment_sample['dxa_img']
    #     mri_img = unaugment_sample['mri_img']
    #     plt.title('MRI')
    #     plt.subplot(221)
    #     plt.imshow(mri_img[0], cmap='gray')
    #     plt.subplot(222)
    #     plt.imshow(mri_img[1], cmap='gray')
    #     plt.subplot(223)
    #     plt.imshow(dxa_img[0], cmap='gray')
    #     plt.subplot(224)
    #     plt.imshow(dxa_img[1], cmap='gray')
    #     plt.show()
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

