import sys, os, glob
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
import torch
import json
import numpy as np
sys.path.append('/users/rhydian/self-supervised-project')
from datasets.BothMidCoronalDataset import BothMidCoronalDataset
from torch.utils.data import Dataset
from registration_model import vgg16_bn


class RegistrationTrainingDataset(Dataset):
    def __init__(self,
                 all_root='/scratch/shared/beegfs/rhydian/UKBiobank/',
                 images_path='/scratch/shared/beegfs/rhydian/UKBiobank/registration_refinement_images/',
                 set_type ='train',
                 augment_first=False,
                 rot_high=10,
                 rot_low=10,
                 trans_high=10,
                 trans_low=10,
                 aligned_scans_names='aligned_scans.txt'):
        super().__init__()
        assert set_type in ['train','val','test']
        self.images_path = os.path.join(images_path,set_type)
        if set_type == 'train':
            with open(aligned_scans_names, 'r') as f:

                self.image_names= [x.replace('\n','') for x in f.readlines()]
        else:
            self.image_names = os.listdir(self.images_path)
        # augmentations for img2
        self.rotation_high = rot_high
        self.translation_high = trans_high
        # augmentaitons for img1
        if augment_first:
            self.rotation_low = rot_low
        else:
            self.rotation_low = 0
        if augment_first:
            self.translation_low = trans_low
        else:
            self.translation_low = 0
        ZERO_PARAM = 0.1

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_path, self.image_names[idx])
        dxas1, mris1 = torch.load(img_path)
        dxas1 = torch.Tensor(dxas1).permute(2,0,1)
        dxas1 = dxas1[:,:mris1.shape[-2],:mris1.shape[-1]]
        img1 = torch.cat([dxas1,mris1])


        delta_angle_1 = self.rotation_low*(np.random.random()-0.5)
        delta_angle_2 = self.rotation_high*(np.random.random()-0.5)
        delta_angle_3 = self.rotation_low*(np.random.random()-0.5)

        translation_1 = [self.translation_low*(np.random.random()-0.5),  self.translation_low*(np.random.random()-0.5)]
        translation_2 = [self.translation_high*(np.random.random()-0.5), self.translation_high*(np.random.random()-0.5)]
        translation_3 = [self.translation_low*(np.random.random()-0.5),  self.translation_low*(np.random.random()-0.5)]

        # transform img1
        img1 = TF.affine(img1[None],delta_angle_1,translation_1,1,[0,0])[0]
        dxas1 = img1[:2]
        mris1 = img1[2:]
        # transform dxa and make img2
        dxas2= TF.affine(dxas1[None],delta_angle_2,translation_2,1,[0,0])[0]
        img2 = torch.cat([dxas2,mris1])

        img2 = TF.affine(img2[None],delta_angle_3,translation_3,1,[0,0])[0]

        return {'img1':img1, 'img2':img2, 'angle1':delta_angle_1,
                'angle2':delta_angle_2,'translations1':torch.Tensor(translation_1), 'translations2':torch.Tensor(translation_2)}

    def __len__(self):
        return len(self.image_names)


def make_trimmed_dataset(ds,out_path='aligned_scans.txt'):

    import matplotlib.pyplot as plt
    ds = RegistrationTrainingDataset(set_type='train',augment_first=False)
    f = open(out_path,'a+')
    while True:
        idx = np.random.randint(len(ds))-1
        sample = ds[idx]
        print('scans so far')
        os.system(f'cat {out_path} | wc')
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.imshow(red(sample['img1'][0])+grayscale(sample['img1'][2]))
        plt.savefig('test.png')
        os.system('imgcat test.png')
        plt.close('all')
        response = None
        while response not in ['y','n','q']:
            response = input('Matches? (y,n,q)')
            if response not in ['y','n','q']:
                print('Error, please type y,n, or q')
        if response == 'q':
            break
        elif response == 'y':
            line = ds.image_names[idx]+'\n'
            if line not in f.readlines():
                f.write(ds.image_names[idx]+'\n')
            else:
                print('already knew that one')
        else:
            continue

    f.close()
    pass

if __name__ == '__main__':
    from gen_utils import *
    ds = RegistrationTrainingDataset(set_type='train',augment_first=True)
    # make_trimmed_dataset(ds)
    model = vgg16_bn(num_classes=3)
    for sample in ds:
        import matplotlib.pyplot as plt
        from gen_utils import *
        out1 = model(sample['img1'][None])
        out2 = model(sample['img2'][None])
        print(out1.shape)
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.imshow(red(sample['img1'][0])+grayscale(sample['img1'][2]))
        plt.subplot(122)
        plt.imshow(red(sample['img2'][0])+grayscale(sample['img2'][2]))
        plt.title(f"{sample['translations2']},{sample['angle2']}")
        plt.savefig('test.png')
        os.system('imgcat test.png')


