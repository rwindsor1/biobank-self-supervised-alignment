import sys, os, glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import json
sys.path.append('/users/rhydian/self-supervised-project')
from datasets.BothMidCoronalDataset import BothMidCoronalDataset


class KeypointAlignmentDataset(BothMidCoronalDataset):
    def __init__(self,
                 all_root='/scratch/shared/beegfs/rhydian/UKBiobank/',
                 annotations_path='/scratch/shared/beegfs/rhydian/UKBiobank/files-for-annotation/'):
        super().__init__(set_type='test',all_root=all_root,crop_scans=True,augment=False,reload_if_fail=True, equal_res=True)
        self.annotations_path = annotations_path
        self.annotations_df = pd.read_csv(os.path.join(annotations_path,'annotation_identifiers.csv'))
        self.dxa_keypoints_file = os.path.join(annotations_path, 'dxa-slice-annotations.json')
        self.scans_df = self.scans_df.iloc[:100]

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        scan_row = self.scans_df.iloc[idx]
        index = self.annotations_df[self.annotations_df['dxa_id']==scan_row['dxa_filename']].index.item()
        mri_keypoints_file = glob.glob(os.path.join(self.annotations_path, 'mri-keypoint-annotations',f'{str(index).zfill(3)}*'))[0]
        # get mri keypoints
        tree = ET.parse(mri_keypoints_file)
        root = tree.getroot()
        point1s = []
        point2s = []
        for entry in root.iter('entry'):
            if 'Point1' == entry.attrib['key']:
                point1s.append(np.array(entry.attrib['value'].split(' '),dtype=np.float)[[0,2]])
            if 'Point2' == entry.attrib['key']:
                point2s.append(np.array(entry.attrib['value'].split(' '),dtype=np.float)[[0,2]])


        assert point1s.__len__() == 5
        assert point2s.__len__() == 5
        mri_keypoints = (np.array(point1s) + np.array(point2s))/2
        mri_keypoints = sorted(mri_keypoints.tolist(),key=lambda x:x[0])
        if mri_keypoints[0][1] > mri_keypoints[1][1]: mri_keypoints[0], mri_keypoints[1] = mri_keypoints[1], mri_keypoints[0]
        if mri_keypoints[-2][1] > mri_keypoints[-1][1]: mri_keypoints[-1], mri_keypoints[-2] = mri_keypoints[-2], mri_keypoints[-1]

        # get dxa keypoints
        with open(self.dxa_keypoints_file,'r') as f:
            json_file = json.load(f)

        for key,val in json_file['file'].items():
            if scan_row['dxa_filename'] in val['fname']:
                fid = val['fid']

        dxa_keypoints = np.array([np.array(val['xy'])[[2,1]] for key, val in json_file['metadata'].items() if int(key.split('_')[0] == fid)])
        dxa_keypoints = sorted(dxa_keypoints.tolist(),key=lambda x:x[0])
        if dxa_keypoints[0][1] > dxa_keypoints[1][1]: dxa_keypoints[0], dxa_keypoints[1] = dxa_keypoints[1], dxa_keypoints[0]
        if dxa_keypoints[-2][1] > dxa_keypoints[-1][1]: dxa_keypoints[-1], dxa_keypoints[-2] = dxa_keypoints[-2], dxa_keypoints[-1]

        # for key, val in json_file['']
        sample['mri_keypoints'] = mri_keypoints
        sample['dxa_keypoints'] = dxa_keypoints


        return sample


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    ds = KeypointAlignmentDataset()
    for idx, sample in enumerate(tqdm(ds)):
        plt.subplot(121)
        plt.imshow(sample['mri_img'][0],cmap='gray')
        for point in sample['mri_keypoints']:
            plt.scatter(point[1],point[0])
        plt.subplot(122)
        plt.imshow(sample['dxa_img'][0],cmap='gray')
        for point in sample['dxa_keypoints']:
            plt.scatter(point[1],point[0])
        plt.show()

