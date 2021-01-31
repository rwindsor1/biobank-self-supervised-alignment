import sys,os,glob
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle

from models.SpatialVGGM import SpatialVGGM
from models.SpatialVGGM2 import SpatialVGGM2
from datasets.MidCoronalDataset import MidCoronalDataset 
from datasets.BothMidCoronalDataset import BothMidCoronalDataset 
from gen_utils import balanced_l1w_loss, grayscale, red
from loss_functions import ContrastiveLoss
from sacred import Experiment
from sacred.observers import MongoObserver
from train_contrastive import get_dataloaders, get_embeds_statistics
import sklearn.decomposition


def fit_pca(ds, dxa_model, mri_model, num_samples = 50, use_cached=False):
    pca_path = 'temp/pca.pkl'
    if os.path.exists(pca_path) and use_cached:
        print(f'Loading PCA from {pca_path}')
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)

    else:
        pca = sklearn.decomposition.PCA(n_components=3)
        all_dxa_spatial_embeds = []
        all_mri_spatial_embeds = []
        pbar = tqdm(total=50)
        print('Calculating PCA...')
        for idx, sample in enumerate(test_ds):
            if idx >= num_samples: break
            with torch.no_grad():
                # dxa_spatial_embeds = F.normalize(dxa_model.module.transformed_spatial_embeddings(sample['dxa_img'][None].cuda()),dim=1)
                # mri_spatial_embeds = F.normalize(mri_model.module.transformed_spatial_embeddings(sample['mri_img'][None].cuda()),dim=1)
                dxa_spatial_embeds = dxa_model.module.transformed_spatial_embeddings(sample['dxa_img'][None].cuda())
                mri_spatial_embeds = mri_model.module.transformed_spatial_embeddings(sample['mri_img'][None].cuda())
            all_dxa_spatial_embeds.append(dxa_spatial_embeds)
            all_mri_spatial_embeds.append(mri_spatial_embeds)
            pbar.update(1)

        dxa_es = torch.cat([x[0].view(128,-1) for x in all_dxa_spatial_embeds],dim=-1)
        mri_es = torch.cat([x[0].view(128,-1) for x in all_mri_spatial_embeds],dim=-1)
        all_es = np.swapaxes(torch.cat([dxa_es.cpu(), mri_es.cpu()], dim=-1).numpy(),0,1)
        pca.fit(all_es)
        print('Saving PCA...')
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
    return pca

def pca_transform(pca, spatial_embeds, resample_size=None):
    shape = spatial_embeds[0,0].shape
    embeddings = np.swapaxes(spatial_embeds[0].view(128,-1).numpy(),0,1)
    transformed_embeddings = pca.transform(embeddings)
    transformed_embeddings = transformed_embeddings.reshape((shape[0], shape[1], 3))
    if resample_size is not None:
        transformed_embeddings = F.interpolate(torch.einsum('ijk->kij',torch.tensor(transformed_embeddings))[None],size=resample_size, mode='bilinear').numpy()[0]
        transformed_embeddings = np.einsum('kij->ijk', transformed_embeddings)
    return transformed_embeddings

def load_models(LOAD_FROM_PATH, POOLING_STRATEGY, USE_FOUR_MODES, model_type):
    # TODO: Implement model loading/saving capacity
    if USE_FOUR_MODES:
        dxa_model = model_type(pooling=POOLING_STRATEGY, input_modes=2)
        mri_model = model_type(pooling=POOLING_STRATEGY, input_modes=2)
    else:
        dxa_model = model_type(pooling=POOLING_STRATEGY)
        mri_model = model_type(pooling=POOLING_STRATEGY)
    dxa_model = nn.DataParallel(dxa_model)
    mri_model = nn.DataParallel(mri_model)
    dxa_model_dict = dxa_model.state_dict()
    mri_model_dict = mri_model.state_dict()
    print(f'Trying to load from {LOAD_FROM_PATH}')
    if not os.path.isdir(LOAD_FROM_PATH):
        os.mkdir(LOAD_FROM_PATH)
    # load model weights
    if os.path.isdir(LOAD_FROM_PATH):
            list_of_pt = glob.glob(LOAD_FROM_PATH + '/*.pt')
            if len(list_of_pt):
                latest_pt = max(list_of_pt, key=os.path.getctime)
                checkpoint = torch.load(latest_pt, map_location=torch.device('cpu'))
                dxa_model_dict.update(checkpoint['dxa_model_weights'])
                mri_model_dict.update(checkpoint['mri_model_weights'])
                dxa_model.load_state_dict(dxa_model_dict)
                mri_model.load_state_dict(mri_model_dict)
                val_stats = checkpoint['val_stats']
                epochs = checkpoint['epochs'] 
                print(f"==> Resuming model trained for {epochs} epochs...")
            else:
                print("==> Training Fresh Model ")
                best_loss = 99999999999999
                epochs = 0
    else:
        raise Exception(f'Could not find directory at {LOAD_FROM_PATH}')

    return dxa_model, mri_model, val_stats, epochs

def val_epoch(dxa_model, mri_model,dl, BATCH_SIZE, MARGIN, USE_INSTANCE_LOSS):
    val_pbar = tqdm(dl)
    dxa_embeds = []
    mri_embeds = []
    val_losses = []
    dxa_model.eval()
    mri_model.eval()
    criterion = ContrastiveLoss(BATCH_SIZE,MARGIN)
    for idx, sample in enumerate(val_pbar):
        dxa_vols = sample['dxa_img']
        mri_vols = sample['mri_img']
        with torch.no_grad():
            dxa_embed = dxa_model(dxa_vols.cuda())
            mri_embed = mri_model(mri_vols.cuda())
            dxa_embed = F.normalize(dxa_embed.squeeze(-1).squeeze(-1),dim=-1)
            mri_embed = F.normalize(mri_embed.squeeze(-1).squeeze(-1),dim=-1)
            if USE_INSTANCE_LOSS:
                loss = criterion(dxa_embed, mri_embed) + criterion(dxa_embed, dxa_embed) + criterion(mri_embed, mri_embed)
            else:
                loss = criterion(dxa_embed, mri_embed)
        val_losses.append(loss.item())
        dxa_embeds.append(dxa_embed.cpu())
        mri_embeds.append(mri_embed.cpu())
        val_pbar.set_description(f"{loss.item():.3}")
    dxa_embeds = torch.cat(dxa_embeds, dim=0)
    mri_embeds = torch.cat(mri_embeds, dim=0)
    val_stats = get_embeds_statistics(dxa_embeds, mri_embeds)
    return val_stats

if __name__ == '__main__':
    DATASET_ROOT='/scratch/shared/beegfs/rhydian/UKBiobank'
    POOLING_STRATEGY='max'
    USE_FOUR_MODES=True
    # LOAD_FROM_PATH ='fully_trained_model_weights/ContrastiveModels_MaxPool_FineTuned'
    LOAD_FROM_PATH ='model_weights/ContrastiveModelsFourModes_MaxPool_TripletLoss'
    if USE_FOUR_MODES:
        model_type = SpatialVGGM
    else:
        model_type = SpatialVGGM2
    if USE_FOUR_MODES:
        test_ds  = BothMidCoronalDataset(set_type='test' ,all_root=DATASET_ROOT, augment=False)
        from train_contrastive_all_modes_negative_mining import get_dataloaders, get_embeds_statistics
        train_dl, val_dl, test_dl = get_dataloaders(10, 10, 10, 10, 10,10, DATASET_ROOT)
    else:
        test_ds  = MidCoronalDataset(set_type='test' ,all_root=DATASET_ROOT,      augment=False)
        from train_contrastive import get_dataloaders, get_embeds_statistics
        train_dl, val_dl, test_dl = get_dataloaders(10, 10, 10, 10, 10,10, False, DATASET_ROOT)
    dxa_model, mri_model, val_stats, epoch_no = load_models(LOAD_FROM_PATH, POOLING_STRATEGY, USE_FOUR_MODES, model_type)

    dxa_model.cuda().eval(); mri_model.cuda().eval()
    test_stats = val_epoch(dxa_model, mri_model, test_dl, 10, 0.1, False)
    print(test_stats)
    pca = fit_pca(test_ds, dxa_model, mri_model, 50)
    pooled_dxas = []
    pooled_mris = []
    for idx, sample in enumerate(tqdm(test_ds)):
        with torch.no_grad():
            # dxa_spatial_embeds = F.normalize(dxa_model.module.transformed_spatial_embeddings(sample['dxa_img'][None].cuda()),dim=1).cpu()
            # mri_spatial_embeds = F.normalize(mri_model.module.transformed_spatial_embeddings(sample['mri_img'][None].cuda()),dim=1).cpu()
            dxa_spatial_embeds = dxa_model.module.transformed_spatial_embeddings(sample['dxa_img'][None].cuda()).cpu()
            mri_spatial_embeds = mri_model.module.transformed_spatial_embeddings(sample['mri_img'][None].cuda()).cpu()
            pooled_dxa = dxa_model.module.pool_function(dxa_spatial_embeds, dxa_spatial_embeds.shape[2:]).cpu()
            pooled_mri = mri_model.module.pool_function(mri_spatial_embeds, mri_spatial_embeds.shape[2:]).cpu()
            pooled_dxas.append(pooled_dxa)
            pooled_mris.append(pooled_mri)
        transformed_dxa_spatial_embeds = pca_transform(pca, dxa_spatial_embeds, resample_size=sample['dxa_img'][0].shape)
        transformed_mri_spatial_embeds = pca_transform(pca, mri_spatial_embeds, resample_size=sample['mri_img'][0].shape)
        plt.figure(figsize=(8,15))
        plt.subplot(221)
        plt.axis('off')
        plt.imshow(sample['dxa_img'][0], cmap='gray')
        plt.subplot(222)
        plt.axis('off')
        plt.imshow(sample['mri_img'][0], cmap='gray')
        plt.subplot(223)
        plt.axis('off')
        plt.imshow(transformed_dxa_spatial_embeds)
        plt.subplot(224)
        plt.axis('off')
        plt.imshow(transformed_mri_spatial_embeds)
        plt.savefig(f'images/mode_invariant_pca_saliency_map/example_saliency_map_{str(idx).zfill(3)}')
        plt.close('all')
        if idx>20:
            break

    import pdb; pdb.set_trace()