import torch
import pickle
import os, sys, glob
sys.path.append('/users/rhydian/self-supervised-project')
from gen_utils import *
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib

class SpineSegmentTrainDataset(Dataset):
    def __init__(self,
                 all_root='/scratch/shared/beegfs/rhydian/UKBiobank',
                 mris_path='Cleaned3DMRIsHalfResolution',
                 segment_path='TransferredMRISegmentations',
                 set_type='train',
                 downsample_scale=2
                ):
        self.mris_path = os.path.join(all_root, mris_path, set_type)
        self.segment_path=os.path.join(all_root,segment_path)
        self.mri_filenames = sorted([x for x in os.listdir(self.mris_path) if 'F.pkl' in x])
        self.downsample_scale=2

    def __len__(self):
        return self.mri_filenames.__len__()

    def __getitem__(self, idx):
        mri_filename = self.mri_filenames[idx]
        # load volume
        with open(os.path.join(self.mris_path, mri_filename),'rb') as f:
            vol = pickle.load(f).float()

        # load segmentation
        segmentation_matches = glob.glob(os.path.join(self.segment_path,'_'.join(mri_filename.split('_')[:-1])+'*'))
        if len(segmentation_matches) != 1:
            #print(f"could not find segmentation at {os.path.join(self.segment_path,'_'.join(mri_filename.split('_')[:-1])+'*')}")
            return self.__getitem__(np.random.randint(self.__len__()))

        segment = torch.load(segmentation_matches[0])[:,:,1]
        vol = vol[None]
        if self.downsample_scale > 1:
            #cvol = F.interpolate(vol[None,None],scale_factor=1/self.downsample_scale, recompute_scale_factor=False).float()[0]
            segment = F.interpolate(segment[None,None],scale_factor=1/self.downsample_scale, recompute_scale_factor=False)[0]

        # check consistent spatial dimensions
        if vol.shape[1] != segment.shape[1] or vol.shape[-1]!=segment.shape[-1]:
            return self.__getitem__(np.random.randint(self.__len__()))


        return {'vol':vol, 'target_2d':segment}

class SpineSegmentTestDataset(Dataset):
    def __init__(self,
                 all_root='/scratch/shared/beegfs/rhydian/UKBiobank',
                 mris_path='mri-niis',
                 segment_path='mri-annotations',
                 annotations_idx='/scratch/shared/beegfs/rhydian/UKBiobank/files-for-annotation/number_subject_matches.txt',
                 set_type='val',
                 downsample_scale=2
                ):
        assert set_type in ['val','test']

        self.mris_path = os.path.join(all_root, mris_path)
        self.segment_path = os.path.join(all_root, segment_path)
        self.set_type=set_type
        self.downsample_scale=downsample_scale
        #self.mri_filenames = os.listdir(self.mris_path)
        # remove water scans
        self.annotations_indexes = {}
        with open(annotations_idx,'r') as f:
            for line in f.readlines():
                idx = int(line.split('_')[0])
                if set_type =='val':
                    if idx>=50:
                        continue
                    else:
                        mri_filename = line[4:-1]
                        self.annotations_indexes[mri_filename] = idx
                else:
                    if idx < 50:
                        continue
                    else:
                        mri_filename = line[4:-1]
                        self.annotations_indexes[mri_filename] = idx


    def __len__(self):
        return len(self.annotations_indexes)


    def __getitem__(self, idx):
        mri_filename = list(self.annotations_indexes.keys())[idx]
        vol = torch.Tensor(nib.load(os.path.join(self.mris_path, mri_filename)).get_fdata().astype(np.float32))
        annotations = torch.Tensor((nib.load(os.path.join(self.segment_path, str(self.annotations_indexes[mri_filename])+'.nii.gz')).get_fdata()>0.5).astype(np.float32))

        if self.downsample_scale > 1:
            vol = F.interpolate(vol[None,None],scale_factor=1/self.downsample_scale, recompute_scale_factor=False)[0]
            annotations = F.interpolate(annotations[None,None],scale_factor=1/self.downsample_scale, recompute_scale_factor=False)[0]
        import pdb; pdb.set_trace()


        return {'vol':torch.Tensor(vol).float(), 'target_3d':torch.Tensor(annotations).float()}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #ds = SpineSegmentTrainDataset()
    val_ds = SpineSegmentTestDataset(set_type='val')
    test_ds = SpineSegmentTestDataset(set_type='test')
    print(len(val_ds))
    print(len(test_ds))
    for sample in val_ds:
        #plt.imshow(grayscale(sample[0][:,int(sample[0].shape[-2]/2),:]) + red(sample[1][:,:]))
        plt.subplot(121)
        plt.imshow(grayscale(sample['vol'][0,:,int(sample['vol'].shape[-2]/2),:])+red(sample['target_3d'][0].sum(axis=-2)))
        plt.subplot(122)
        plt.imshow(grayscale(sample['vol'][0,:,:,int(sample['vol'].shape[-1]/2)])+red(sample['target_3d'][0].sum(axis=-1)))
        plt.savefig('test.png')
        os.system('imgcat test.png')


