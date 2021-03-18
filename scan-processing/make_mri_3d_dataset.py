import os, sys, glob
import csv
import pickle
from shutil import copyfile
from tqdm import tqdm
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--frac', type=int)
parser.add_argument('--num_fracs', type=int, default=20)

args = parser.parse_args()

scans_path = '/scratch/shared/beegfs/rhydian/UKBiobank/stitched_mris'
scan_list_dir = '/users/rhydian/self-supervised-project/scan_lists/'
out_dir = '/scratch/shared/beegfs/rhydian/UKBiobank/Cleaned3DMRIs'



# read in scan info csv
scans_info = {}
for set_ in ['train','test','val']:
    scan_list_path = os.path.join(scan_list_dir, f'biobank_mri_dxa_id_{set_}.csv')
    set_info_dict = {}
    with open(scan_list_path,'r') as f:
        reader = csv.reader(f)
        for idx, rows in enumerate(reader):
            if idx == 0:
                for key in rows:
                    set_info_dict[key] = []
            else:
                for idx, key in enumerate(set_info_dict):
                    set_info_dict[key].append(rows[idx])

    scans_info[set_] = set_info_dict



scans = os.listdir(scans_path)

# get scans
print(scans_info['train'].keys())
for key in scans_info:
    if not os.path.isdir(os.path.join(out_dir,key)):
        os.mkdir(os.path.join(out_dir,key))

    print(f"Copying scans for {key} set/...")
    num_scans = len(scans_info[key]['mri_filename'])
    START_IDX = int(np.floor(num_scans*(args.frac)/args.num_fracs))
    END_IDX = int(np.ceil(num_scans*(args.frac+1)/args.num_fracs))
    if END_IDX >= num_scans: END_IDX = num_scans - 1
    print(f'Going from idx {START_IDX} to idx {END_IDX} from a list of len {num_scans}')

    for idx, mri_filename in enumerate(tqdm(scans_info[key]['mri_filename'][START_IDX:END_IDX])):
        scan_path_root = os.path.join(scans_path,mri_filename)
        scans = glob.glob(scan_path_root+'*')
        fat_scan = None; water_scan = None
        for scan in scans:
            if '_F.pkl' in scan: fat_scan = scan
            if '_W.pkl' in scan: water_scan = scan
            if (fat_scan != None) and (water_scan != None):
                out_fat_scan_path = os.path.join(out_dir, key, fat_scan.split('/')[-1])
                out_water_scan_path = os.path.join(out_dir, key, water_scan.split('/')[-1])
                if not os.path.isfile(out_fat_scan_path):
                    copyfile(fat_scan, out_fat_scan_path)
                if not os.path.isfile(out_water_scan_path):
                    copyfile(water_scan, out_water_scan_path)








print(scans[0])


