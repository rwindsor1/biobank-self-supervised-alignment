import sys, os, glob
import pickle

from scipy.io import loadmat
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from shapely.geometry import Polygon
import ast

class MRISlicesSegmentationDataset(Dataset):
    '''
    Dataset class for stitched MRI scans with associated detected vertebrae
    '''
    def __init__(self, 
                 all_root='/tmp/rhydian/',
                 aligned_scans_file='/users/rhydian/ukbb-dxa-mri-project/scan_lists/',
                 stitched_scans_path = '/tmp/rhydian/SynthesisedMRISlices/',
                 dxa_root = '/tmp/rhydian/dxa/bone',
                set_type = 'train',
                 augment = True,
                 blanking_augment=False):

        super(MRISlicesSegmentationDataset, self).__init__()
        assert set_type in ['train', 'val', 'test', 'all'], f"Allowed set types {set_type}"
        if set_type == 'all': aligned_scans_list = os.path.join(aligned_scans_file, 'biobank_mri_dxa_id_alignments.csv') 
        else:
            aligned_scans_list = os.path.join(aligned_scans_file, f'biobank_mri_dxa_id_alignments_{set_type}.csv')
        self.set_type = set_type
        self.aligned_scan_df = pd.read_csv(aligned_scans_list.replace('/tmp/rhydian/',all_root))
        self.scans_path = stitched_scans_path.replace('/tmp/rhydian/',all_root)
        self.dxa_root = dxa_root.replace('/tmp/rhydian/',all_root)
        self.augment = augment
        self.curvature_df = pd.read_csv('/users/rhydian/ukbb-dxa-mri-project/ukbb_dxa_curvature_v3.csv')
        self.segmentation_dir = '/tmp/rhydian/dxa-segmentation3'.replace('/tmp/rhydian/',all_root)
        self.blanking_augment = blanking_augment

    
    def __len__(self):
        return self.aligned_scan_df.shape[0]

    def get_uncropped_images(self, idx):
        scan_row = self.aligned_scan_df.iloc[idx]
        dxa_path = os.path.join(self.dxa_root, scan_row['dxa_filename'] + '.mat')
        scan_name = os.path.join(self.scans_path, self.set_type, scan_row['mri_filename'] + '_F')
        mri_volume = torch.load(scan_name)[None,None]
        dxa_volume = torch.Tensor(loadmat(dxa_path)['scan'])[None,None]
        dxa_segmentation = torch.Tensor(np.load(os.path.join(self.segmentation_dir, scan_row['dxa_filename']+'.npy')))[:,:,:].permute(2,0,1)
        dxa_segmentation = ((dxa_segmentation != 0).int())[None]
        return {'mri_vol':mri_volume, 'dxa_vol':dxa_volume,
               'mri_id':  scan_row['mri_filename'], 'dxa_id':scan_row['dxa_filename']}

    def find_by_dxa_id(self, dxa_id):
        idxs = self.aligned_scan_df[self.aligned_scan_df['dxa_filename'] == dxa_id].index
        if len(idxs):
            return self.__getitem__(idxs[0])
        else:
            return None

    def __getitem__(self, idx):
        try:
            scan_row = self.aligned_scan_df.iloc[idx]
            dxa_path = os.path.join(self.dxa_root, scan_row['dxa_filename'] + '.mat')
            scan_name = os.path.join(self.scans_path, self.set_type, scan_row['mri_filename'] + '_F')
            # parameters for augmentation
            ROT_LOW = -10
            ROT_HIGH = 10
            TRANS_LOW = -4
            TRANS_HIGH = 5
            if self.augment:
                mri_rot     = np.random.randint(ROT_LOW,ROT_HIGH)
                dxa_rot     = np.random.randint(ROT_LOW,ROT_HIGH)
                dxa_delta_x = np.random.randint(TRANS_LOW,TRANS_HIGH) 
                dxa_delta_y = np.random.randint(TRANS_LOW,TRANS_HIGH)
                mri_delta_x = np.random.randint(TRANS_LOW,TRANS_HIGH)
                mri_delta_y = np.random.randint(TRANS_LOW,TRANS_HIGH)
            else:
                mri_rot = 0; dxa_rot = 0; dxa_delta_x = 0; dxa_delta_y = 0; mri_delta_x = 0; mri_delta_y = 0

            mri_volume = torch.load(scan_name)[None,None]


            dxa_volume = torch.Tensor(loadmat(dxa_path)['scan'])[None,None]
            dxa_segmentation = torch.Tensor(np.load(os.path.join(self.segmentation_dir, scan_row['dxa_filename']+'.npy')))[:,:,:].permute(2,0,1)
            dxa_segmentation = ((dxa_segmentation != 0).int())[None]

            if mri_rot != 0:
                mri_volume = TF.rotate(mri_volume, mri_rot)
            if dxa_rot != 0:
                dxa_volume = TF.rotate(dxa_volume, dxa_rot)
                dxa_segmentation = TF.rotate(dxa_segmentation, dxa_rot)

            dxa_volume = F.pad(dxa_volume,[-dxa_delta_x, dxa_delta_x, -dxa_delta_y, dxa_delta_y])[0,0]
            dxa_segmentation = F.pad(dxa_segmentation,[-dxa_delta_x, dxa_delta_x, -dxa_delta_y, dxa_delta_y])[0]
            mri_volume = F.pad(mri_volume,[-mri_delta_x, mri_delta_x,-mri_delta_y, mri_delta_y])[0,0]
            
        
            output_dxa_shape = (700,276)
            output_mri_shape = (501,224)
            if dxa_volume.shape[0] != output_dxa_shape[0] or dxa_volume.shape[1] != output_dxa_shape[1]:
                diff = (output_dxa_shape[0] - dxa_volume.shape[0],output_dxa_shape[1] - dxa_volume.shape[1])
                dxa_volume = F.pad(dxa_volume, [int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])
            if dxa_segmentation.shape[1] != output_dxa_shape[0] or dxa_segmentation.shape[2] != output_dxa_shape[1]:
                dxa_segmentation = F.pad(dxa_segmentation,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])
            if mri_volume.shape[0] != output_mri_shape[0] or mri_volume.shape[1] != output_mri_shape[1]:
                diff = (output_mri_shape[0] - mri_volume.shape[0],output_mri_shape[1] - mri_volume.shape[1])
                mri_volume = F.pad(mri_volume,[int(np.floor(diff[1]/2)),int(np.ceil(diff[1]/2)),int(np.floor(diff[0]/2)),int(np.ceil(diff[0]/2))])

            assert (dxa_volume.shape[0] == 700 and dxa_volume.shape[1] == 276), f"dxa_volume {dxa_volume.shape}"
            assert (dxa_segmentation.shape[1] == 700 and dxa_segmentation.shape[2] == 276), f"dxa_segmentation {dxa_segmentation.shape}"
            assert (mri_volume.shape[0] == 501 and mri_volume.shape[1] == 224), f"mri_volume {mri_volume.shape}"

            if self.blanking_augment:
                if np.random.random() > 0.25:
                    rand_h = np.random.randint(np.ceil(dxa_volume.shape[0]/5),np.ceil(dxa_volume.shape[0]/4))
                    rand_x = np.random.randint(0, dxa_volume.shape[0] - rand_h)
                    dxa_volume[rand_x:rand_x+rand_h] = 0

                if np.random.random() > 0.25:
                    rand_h = np.random.randint(np.ceil(mri_volume.shape[0]/5),np.ceil(mri_volume.shape[0]/4))
                    rand_x = np.random.randint(0, mri_volume.shape[0] - rand_h)
                    mri_volume[rand_x:rand_x+rand_h] = 0

            return {'mri_vol': mri_volume[None].float(), 'dxa_vol': dxa_volume[None].float(), 
                    'mri_id':  scan_row['mri_filename'], 'dxa_id':scan_row['dxa_filename'],
                    'dxa_segmentation': dxa_segmentation}

        except Exception as e:
            rand_idx = np.random.randint(self.__len__())
            return self.__getitem__(rand_idx)


if __name__ == '__main__':
    import warnings; warnings.warn(f"Running {__file__} in test mode!")
    from torch.utils.data import DataLoader
    import tqdm
    import matplotlib
    sys.path.append('/users/rhydian/self-supervised-project')
    from gen_utils import red, blue, green, pink, yellow, brown, grayscale
    import pandas as pd
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid, save_image
    df = pd.read_csv('/users/rhydian/ukbb-dxa-mri-project/dice_results.csv').sort_values('avg')

    ds = MRISlicesSegmentationDataset(all_root='/scratch/shared/beegfs/rhydian/UKBiobank/',augment=False,blanking_augment=False)
    for idx, sample in enumerate(ds):
        print(sample['dxa_segmentation'].shape)
        plt.title(sample['dxa_id'])
        image = grayscale(sample['dxa_vol'][0])
        image += red( sample['dxa_segmentation'][0]) # skull 
        image += green( sample['dxa_segmentation'][1]) # spine
        image += blue(  sample['dxa_segmentation'][2]) # pelvis
        image += pink(  sample['dxa_segmentation'][3]) # right leg
        image += yellow(sample['dxa_segmentation'][4]) # left leg
        image += brown( sample['dxa_segmentation'][5]) # pelvic cavity
        plt.imshow(image)
        plt.show()

