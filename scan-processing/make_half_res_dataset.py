import os, glob, sys
import pickle
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start_frac', type=float, help='The fraction of the dataset to start at')
parser.add_argument('--end_frac',   type=float, help='The fraction of the dataset to end at')
parser.add_argument('--set_type',   type=str, help='The dataset to run over')
args=parser.parse_args()



FILES_PATH = '/scratch/shared/beegfs/rhydian/UKBiobank/Cleaned3DMRIsHalfPrecision'
OUT_PATH = '/scratch/shared/beegfs/rhydian/UKBiobank/Cleaned3DMRIsHalfResolution'
SET_TYPE = args.set_type

if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)
if not os.path.isdir(os.path.join(OUT_PATH, SET_TYPE)):
    os.mkdir(os.path.join(OUT_PATH,SET_TYPE))

all_files = sorted(os.listdir(os.path.join(FILES_PATH,SET_TYPE)))

start_idx = int(np.floor(len(all_files)*args.start_frac))
end_idx =   int(np.floor(len(all_files)*args.end_frac) - 1)
for idx in tqdm(range(start_idx,end_idx)):
    file = all_files[idx]
    if 'W.pkl' in file:
        continue

    with open(os.path.join(FILES_PATH, SET_TYPE, file),'rb') as f:
        scan = pickle.load(f)
    new_scan = F.interpolate(scan.float()[None,None],scale_factor=0.5, recompute_scale_factor=False).half()[0,0]
    with open(os.path.join(OUT_PATH, SET_TYPE, file),'wb') as f:
        pickle.dump(new_scan,f)


