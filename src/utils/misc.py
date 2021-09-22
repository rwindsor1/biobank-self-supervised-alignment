from torch.serialization import load
from functools import cmp_to_key
from pickle import load
from typing import OrderedDict
import torch
from torch import optim
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import glob
import os
import re
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

def optimiser_to(optim : torch.optim.Optimizer, 
                 device: torch.device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def load_checkpoint(dxa_model : nn.Module, 
                        mri_model: nn.Module, 
                        optimiser : torch.optim.Optimizer,
                        load_from_path : str,
                        use_cuda: bool,
                        verbose : bool=True):

    if not os.path.isdir(load_from_path):
        os.mkdir(load_from_path)
    # load model weights
    if os.path.isdir(load_from_path):
            list_of_pt = glob.glob(load_from_path + '/*.pt')
            if len(list_of_pt):
                dxa_model_dict = dxa_model.state_dict()
                mri_model_dict = mri_model.state_dict()
                optim_state_dict = optimiser.state_dict()
                latest_pt = max(list_of_pt, key=os.path.getctime)
                print(latest_pt)
                checkpoint = torch.load(latest_pt, map_location=torch.device('cpu'))
                # alter model weight names if they have been saved from nn.DataParallel object
                for model_weights in ['dxa_model_weights','mri_model_weights']:
                    checkpoint[model_weights] = OrderedDict((re.sub('^module\.','',k) 
                                                if re.search('^module.',k) else k, v) 
                                                for k,v in checkpoint[model_weights].items())
                        
                dxa_model_dict.update(checkpoint['dxa_model_weights'])
                mri_model_dict.update(checkpoint['mri_model_weights'])
                
                dxa_model.load_state_dict(dxa_model_dict)
                mri_model.load_state_dict(mri_model_dict)
                if 'optim_state' in checkpoint:
                    optim_state_dict.update(checkpoint['optim_state'])
                    optimiser.load_state_dict(optim_state_dict)
                val_stats = checkpoint['val_stats']
                epochs = checkpoint['epochs'] 
                if verbose:
                    print(f"==> Resuming model trained for {epochs} epochs...")
            else:
                if verbose:
                    print("==> Training Fresh Model ")
                val_stats = {'mean_rank':999999999999}
                epochs = 0
    if use_cuda:
        dxa_model.to('cuda:0')
        mri_model.to('cuda:0')
        optimiser_to(optimiser,'cuda:0')

    return dxa_model, mri_model, optimiser,val_stats, epochs 


def save_checkpoint(dxa_model : nn.Module, mri_model : nn.Module,
                    optimiser : torch.optim.Optimizer, 
                    val_stats : dict, 
                    epochs : int, save_weights_path : str):
    if isinstance(dxa_model, nn.DataParallel): dxa_model = dxa_model.module
    if isinstance(mri_model, nn.DataParallel): mri_model = mri_model.module
    print(f'==> Saving Model Weights to {save_weights_path}')
    state = {'dxa_model_weights': dxa_model.state_dict(),
             'mri_model_weights': mri_model.state_dict(),
             'optim_state'      : optimiser.state_dict(),
             'val_stats'        : val_stats,
             'epochs'           : epochs
            }
    if not os.path.isdir(save_weights_path):
        os.mkdir(save_weights_path)
    previous_checkpoints = glob.glob(save_weights_path + '/ckpt*.pt', recursive=True)
    torch.save(state, save_weights_path + '/ckpt' + str(epochs) + '.pt')
    for previous_checkpoint in previous_checkpoints:
       os.remove(previous_checkpoint)
    return


def get_batch_corrrelations(scan_embeds_1, scan_embeds_2):
    ''' gets correlations between scan embeddings'''
    batch_size, channels, h, w = scan_embeds_2.shape

    scan_embeds_1 = F.normalize(scan_embeds_1,dim=1)
    scan_embeds_2 = F.normalize(scan_embeds_2,dim=1)
    correlation_maps = F.conv2d(scan_embeds_1, scan_embeds_2)/(h*w)
    return correlation_maps

def get_dataset_similarities(scan_embeds_1, scan_embeds_2, batch_size=50):
    ''' Gets similarities for entire dataset. 
    Splits job into batches to reduce GPU memory'''
    ds_size, channels, h, w = scan_embeds_2.shape
    ds_similarities = torch.zeros(ds_size, ds_size)
   
    for batch_1_start_idx in tqdm(range(0, ds_size, batch_size)):
        for batch_2_start_idx in range(0, ds_size, batch_size):

            batch_1_end_idx = batch_1_start_idx + batch_size
            batch_2_end_idx = batch_2_start_idx + batch_size
            if batch_2_end_idx >= ds_size: batch_2_end_idx = ds_size
            if batch_1_end_idx >= ds_size: batch_1_end_idx = ds_size
            
            correlations = get_batch_corrrelations(scan_embeds_1[batch_1_start_idx:batch_1_end_idx],
                                                   scan_embeds_2[batch_2_start_idx:batch_2_end_idx])
            similarities,_ = torch.max(correlations.flatten(start_dim=2),dim=-1)
            ds_similarities[batch_1_start_idx:batch_1_end_idx,batch_2_start_idx:batch_2_end_idx] = similarities
    return ds_similarities