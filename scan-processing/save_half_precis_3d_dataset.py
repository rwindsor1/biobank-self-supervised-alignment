import sys, os, glob
import pickle
import torch
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--frac', type=int)
parser.add_argument('--num_fracs', type=int, default=20)
args = parser.parse_args()

dataset_root = '/scratch/shared/beegfs/rhydian/UKBiobank/Cleaned3DMRIs'
output_root =  '/scratch/shared/beegfs/rhydian/UKBiobank/Cleaned3DMRIsHalfPrecision'


for set_type in ['train','test','val']:
    dataset_path = os.path.join(dataset_root,set_type)
    output_path =  os.path.join(output_root, set_type)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    filenames = os.listdir(dataset_path)
    num_scans = filenames.__len__()
    START_IDX = int(np.floor(num_scans*(args.frac)/args.num_fracs))
    END_IDX = int(np.ceil(num_scans*(args.frac+1)/args.num_fracs))

    for filename in tqdm(filenames[START_IDX:END_IDX]):
        with open(os.path.join(dataset_path, filename),'rb') as f:
            scan = pickle.load(f)
        volume = scan['volume']

        volume_hlf_pre = torch.Tensor(scan['volume']).half()
        with open(os.path.join(output_path,filename),'wb') as f:
            pickle.dump(volume_hlf_pre,f)


