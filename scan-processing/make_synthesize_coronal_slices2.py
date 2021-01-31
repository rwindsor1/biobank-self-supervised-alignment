''' Script to extract the mid sagittal slices from the MRI arrays based on where the spine is detected '''

import os, sys, glob
import pickle
sys.path.append('/users/rhydian/self-supervised-project')
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from gen_utils import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--start_frac', type=float, help='The fraction of the dataset to start at')
parser.add_argument('--end_frac',   type=float, help='The fraction of the dataset to end at')
parser.add_argument('--set_type',   type=str, help='The dataset to run over')

def make_synthesized_slices(volume, target):
    target_points = get_points(target, threshold=0.5)
    target_points.sort(key=lambda x: x[-1])
    mid_sagittal_slices = np.zeros(volume.shape[0])
    synthesized_img = torch.zeros(volume.shape[0], volume.shape[2])
    for scan_line_idx in range(mid_sagittal_slices.shape[0]):
        closest_points = sorted([(point, np.abs(scan_line_idx-point[0])) for point in target_points],key=lambda x: x[1])[:2]
        sag_slice = (closest_points[0][0][1]*closest_points[1][1]+closest_points[1][0][1]*closest_points[0][1])/(closest_points[0][1]+closest_points[1][1])
        mid_sagittal_slices[scan_line_idx] = int(sag_slice) 
        synthesized_img[scan_line_idx] = volume[scan_line_idx, int(sag_slice),:]

    return synthesized_img

def main(args):
    stitched_scans_path = '/scratch/shared/beegfs/rhydian/UKBiobank/stitched_mris/'
    targets_path = '/scratch/shared/beegfs/rhydian/UKBiobank/FullResMRIArrays/'
    synthesized_mri_slice_path = '/scratch/shared/beegfs/rhydian/UKBiobank/SynthesisedMRISlices2/'
    dxa_root = '/scratch/shared/beegfs/rhydian/UKBiobank/dxa/'
    aligned_scans_path = '/users/rhydian/self-supervised-project/scan_lists'
    errors = 'failed_scans.txt'
    set_type = args.set_type
    if set_type == 'all': aligned_scans_list = os.path.join(aligned_scans_path, 'biobank_mri_dxa_id_alignments.csv')
    else: aligned_scans_list = os.path.join(aligned_scans_path, f'biobank_mri_dxa_id_alignments_{set_type}.csv')

    aligned_scans_list = pd.read_csv(aligned_scans_list)
    start_idx = int(np.floor(len(aligned_scans_list)*args.start_frac))
    end_idx =   int(np.floor(len(aligned_scans_list)*args.end_frac) - 1)

    if not os.path.isdir(os.path.join(synthesized_mri_slice_path, set_type)): os.mkdir(os.path.join(synthesized_mri_slice_path, set_type))
    save_path = os.path.join(synthesized_mri_slice_path, set_type)

    for row_idx in tqdm(range(start_idx, end_idx)):

        sample = aligned_scans_list.iloc[row_idx]
        fat_mri_scan_name = sample['mri_filename'] + '_F'
        water_mri_scan_name = sample['mri_filename'] + '_W'
        target_name = sample['mri_filename'] + '_F'
        dxa_scan_name = sample['dxa_filename'] + '.mat'
        try:
            with open(os.path.join(stitched_scans_path, fat_mri_scan_name + '.pkl'),'rb') as f:
                fat_volume = torch.tensor(pickle.load(f)['volume'])
            with open(os.path.join(stitched_scans_path, water_mri_scan_name + '.pkl'),'rb') as f:
                water_volume = torch.tensor(pickle.load(f)['volume'])
            #fat_volume = torch.load(os.path.join(stitched_scans_path, fat_mri_scan_name))
            target = torch.load(os.path.join(targets_path, set_type, target_name, 'target3'))
            # dxa_bone_scan = loadmat(os.path.join(dxa_root,'bone', dxa_scan_name))['scan']
            # dxa_tissue_scan = loadmat(os.path.join(dxa_root,'tissue', dxa_scan_name))['scan']
            synthesized_fat_img = make_synthesized_slices(fat_volume, target)
            synthesized_water_img = make_synthesized_slices(water_volume, target)
            # plt.figure(figsize=(10,10))
            # plt.subplot(221)
            # plt.imshow(synthesized_fat_img, cmap='gray')
            # plt.subplot(222)
            # plt.imshow(synthesized_water_img, cmap='gray')
            # plt.subplot(223)
            # plt.imshow(dxa_bone_scan, cmap='bone')
            # plt.subplot(224)
            # plt.imshow(dxa_tissue_scan, cmap='gray')
            # plt.show()
            torch.save(synthesized_fat_img, os.path.join(save_path, fat_mri_scan_name))
            torch.save(synthesized_water_img, os.path.join(save_path, water_mri_scan_name))
        except Exception as E:
            with open(errors, 'a') as f:
                f.write(fat_mri_scan_name + '\n')

            print(fat_mri_scan_name)
            print(E)


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.set_type in ['train','val','test','all']
    main(args)

