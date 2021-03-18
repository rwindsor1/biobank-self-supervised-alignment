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

ex = Experiment("VisualizeSpatialFeatures")

@ex.config
def config():
    DATASET_ROOT='/scratch/shared/beegfs/rhydian/UKBiobank'
    POOLING_STRATEGY='max'
    USE_FOUR_MODES=False
    LOAD_FROM_PATH ='/users/rhydian/self-supervised-project/model_weights/ContrastiveModelsFromScratch2_MaxPool_TripletLoss'
    NORMALIZE_SPATIAL_FEATURES=True
    MODEL_TYPE = SpatialVGGM # SpatialVGGM, SpatialVGGM2
    NOTE=''
    SAVE_IMAGES=False
    # TEST EPOCH SETTINGS
    MEASURE_TEST_PERFORMANCE = True
    BATCH_SIZE=10
    MARGIN=0.1
    # FIT PCA SETTTINGS
    USE_CACHED_PCA = False 
    NUM_PCA_SAMPLES=50 # number of samples to fit pca on
    OUTPUT_PATH = os.path.join('visualized_spatial_features',NOTE)
    # SAVE PCA FEATURE MAPS SETTINGS
    NUM_PCA_FEATURE_MAPS_SAVED = 20
    MAKE_CORRELATION_MAPS = True

    # MAKE CORRELATION MAPS
    USE_SOFTMAX = True

@ex.capture
def fit_pca(dxa_model, 
            mri_model, ds, 
            USE_CACHED_PCA,
            NORMALIZE_SPATIAL_FEATURES,NUM_PCA_SAMPLES,OUTPUT_PATH):


    pca_path = os.path.join(OUTPUT_PATH, 'pca.pkl')
    if os.path.exists(pca_path) and USE_CACHED_PCA:
        print(f'Loading PCA from {pca_path}')
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)

    else:
        pca = sklearn.decomposition.PCA(n_components=3)
        all_dxa_spatial_embeds = []
        all_mri_spatial_embeds = []
        pbar = tqdm(total=50)
        print('Calculating PCA...')
        for idx, sample in enumerate(ds):
            if idx >= NUM_PCA_SAMPLES: break
            with torch.no_grad():
                if NORMALIZE_SPATIAL_FEATURES:
                    dxa_spatial_embeds = F.normalize(dxa_model.module.transformed_spatial_embeddings(sample['dxa_img'][None].cuda()),dim=1)
                    mri_spatial_embeds = F.normalize(mri_model.module.transformed_spatial_embeddings(sample['mri_img'][None].cuda()),dim=1)
                else:
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

@ex.capture
def pca_transform(pca, spatial_embeds, resample_size=None):
    shape = spatial_embeds[0,0].shape
    embeddings = np.swapaxes(spatial_embeds[0].view(128,-1).numpy(),0,1)
    transformed_embeddings = pca.transform(embeddings)
    transformed_embeddings = transformed_embeddings.reshape((shape[0], shape[1], 3))
    if resample_size is not None:
        transformed_embeddings = F.interpolate(torch.einsum('ijk->kij',torch.tensor(transformed_embeddings))[None],size=resample_size, mode='bilinear',align_corners=False).numpy()[0]
        transformed_embeddings = np.einsum('kij->ijk', transformed_embeddings)
    return transformed_embeddings


@ex.capture
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

@ex.capture
def test_epoch(dxa_model, mri_model,dl, BATCH_SIZE, MARGIN, NORMALIZE_SPATIAL_FEATURES):

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
            dxa_embed = dxa_embed.squeeze(-1).squeeze(-1)
            mri_embed = mri_embed.squeeze(-1).squeeze(-1)
            if NORMALIZE_SPATIAL_FEATURES:
                dxa_embed = F.normalize(dxa_embed, dim=1)
                mri_embed = F.normalize(mri_embed, dim=1)
            loss = criterion(dxa_embed, mri_embed) + criterion(dxa_embed, dxa_embed) + criterion(mri_embed, mri_embed)

        val_losses.append(loss.item())
        dxa_embeds.append(dxa_embed.cpu())
        mri_embeds.append(mri_embed.cpu())
        val_pbar.set_description(f"{loss.item():.3}")
    dxa_embeds = torch.cat(dxa_embeds, dim=0)
    mri_embeds = torch.cat(mri_embeds, dim=0)
    val_stats = get_embeds_statistics(dxa_embeds, mri_embeds)
    return val_stats

@ex.capture
def assess_approximation(dxa_model, mri_model,test_ds, normalize=False, num_scans=50):
    pbar = tqdm(test_ds)
    all_dxa_scan_embeds = []
    all_mri_scan_embeds = []
    all_pooled_dxa_spatial_embeds = []
    all_pooled_mri_spatial_embeds = []
    val_losses = []
    dxa_model.eval()
    mri_model.eval()
    for idx, sample in enumerate(pbar):
        if idx >= num_scans: break
        dxa_vols = sample['dxa_img'][None]
        mri_vols = sample['mri_img'][None]
        with torch.no_grad():
            dxa_scan_embed = dxa_model(dxa_vols.cuda()).squeeze(-1).squeeze(-1).cpu()
            mri_scan_embed = mri_model(mri_vols.cuda()).squeeze(-1).squeeze(-1).cpu()

            dxa_spatial_embeds = dxa_model.module.transformed_spatial_embeddings(sample['dxa_img'][None].cuda()).cpu() 
            mri_spatial_embeds = mri_model.module.transformed_spatial_embeddings(sample['mri_img'][None].cuda()).cpu() 

            pooled_dxa_spatial_embeds = dxa_model.module.pool_function(dxa_spatial_embeds, dxa_spatial_embeds.shape[2:]).squeeze(-1).squeeze(-1)
            pooled_mri_spatial_embeds = mri_model.module.pool_function(mri_spatial_embeds, mri_spatial_embeds.shape[2:]).squeeze(-1).squeeze(-1)


        if normalize:
            dxa_scan_embed = F.normalize(dxa_scan_embed)
            mri_scan_embed = F.normalize(mri_scan_embed)
            pooled_dxa_spatial_embeds = F.normalize(pooled_dxa_spatial_embeds)
            pooled_mri_spatial_embeds = F.normalize(pooled_mri_spatial_embeds)


        all_pooled_dxa_spatial_embeds.append(pooled_dxa_spatial_embeds)
        all_pooled_mri_spatial_embeds.append(pooled_mri_spatial_embeds)
        all_dxa_scan_embeds.append(dxa_scan_embed)
        all_mri_scan_embeds.append(mri_scan_embed)
        

    all_dxa_scan_embeds = torch.cat(all_dxa_scan_embeds)
    all_mri_scan_embeds = torch.cat(all_mri_scan_embeds)
    scan_embed_distances = (all_dxa_scan_embeds[:,None] - all_mri_scan_embeds[None,:]).norm(dim=-1)
    mean_match_scan_embed_distances = scan_embed_distances.diag().mean()
    mean_non_match_scan_embed_distances = scan_embed_distances[~np.eye(scan_embed_distances.shape[0],dtype=bool)].mean()

    all_pooled_dxa_spatial_embeds = torch.cat(all_pooled_dxa_spatial_embeds)
    all_pooled_mri_spatial_embeds = torch.cat(all_pooled_mri_spatial_embeds)
    pooled_spatial_embeds_distances = (all_pooled_dxa_spatial_embeds[:,None] - all_pooled_mri_spatial_embeds[None,:]).norm(dim=-1)
    mean_match_pooled_spatial_embeds_distances = pooled_spatial_embeds_distances.diag().mean()
    mean_non_match_pooled_spatial_embeds_distances = pooled_spatial_embeds_distances[~np.eye(pooled_spatial_embeds_distances.shape[0],dtype=bool)].mean()

    dxa_distances = (all_dxa_scan_embeds[:,None] - all_pooled_dxa_spatial_embeds[None,:]).norm(dim=-1)
    mri_distances = (all_mri_scan_embeds[:,None] - all_pooled_mri_spatial_embeds[None,:]).norm(dim=-1)

    mean_match_dxa_distances = dxa_distances.diag().mean()
    mean_match_mri_distances = mri_distances.diag().mean()
    mean_non_match_dxa_distances = dxa_distances[~np.eye(dxa_distances.shape[0],dtype=bool)].mean()
    mean_non_match_mri_distances = mri_distances[~np.eye(mri_distances.shape[0],dtype=bool)].mean()

    results_dict = {'dxa_distances': {
                        'match':     mean_match_dxa_distances,
                        'non_match': mean_non_match_dxa_distances
                    },
                    'mri_distances': {
                        'match':     mean_match_mri_distances,
                        'non_match': mean_non_match_mri_distances
                    },
                    'scan_embed_distances': {
                        'match':     mean_match_scan_embed_distances,
                        'non_match': mean_non_match_scan_embed_distances
                    },
                    'pooled_spatial_embeds_distances':{
                        'match':     mean_match_pooled_spatial_embeds_distances,
                        'non_match': mean_non_match_pooled_spatial_embeds_distances
                    }}
                
    for key in results_dict:
        print(f"{key}:\nMatch {results_dict[key]['match']:.4}, Non-Match {results_dict[key]['non_match']:.4}")
    import pdb; pdb.set_trace()
        

@ex.capture
def get_distances_matrix(dxa_embeds, mri_embeds):
    dxa_embeds = torch.cat(dxa_embeds)
    mri_embeds = torch.cat(mri_embeds)

    return torch.norm(dxa_embeds[:,None,:] - mri_embeds[None,:,:], dim=-1)

@ex.capture
def save_spatial_feature_maps(dxa_model, mri_model, test_ds,pca, NORMALIZE_SPATIAL_FEATURES, OUTPUT_PATH, NUM_PCA_FEATURE_MAPS_SAVED,MAKE_CORRELATION_MAPS, SAVE_IMAGES):
    pooled_dxas = []
    pooled_mris = []
    for idx, sample in enumerate(tqdm(test_ds)):
        with torch.no_grad():
            if NORMALIZE_SPATIAL_FEATURES:
                dxa_spatial_embeds = F.normalize(dxa_model.module.transformed_spatial_embeddings(sample['dxa_img'][None].cuda()),dim=1).cpu()
                mri_spatial_embeds = F.normalize(mri_model.module.transformed_spatial_embeddings(sample['mri_img'][None].cuda()),dim=1).cpu() 
            else: 
                dxa_spatial_embeds = dxa_model.module.transformed_spatial_embeddings(sample['dxa_img'][None].cuda()).cpu() 
                mri_spatial_embeds = mri_model.module.transformed_spatial_embeddings(sample['mri_img'][None].cuda()).cpu() 
            pooled_dxa = dxa_model.module.pool_function(dxa_spatial_embeds, dxa_spatial_embeds.shape[2:]).cpu() 
            pooled_mri = mri_model.module.pool_function(mri_spatial_embeds, mri_spatial_embeds.shape[2:]).cpu() 
            pooled_dxas.append(pooled_dxa.cpu().squeeze(-1).squeeze(-1)) 
            pooled_mris.append(pooled_mri.cpu().squeeze(-1).squeeze(-1)) 
            transformed_dxa_spatial_embeds = pca_transform(pca, dxa_spatial_embeds, resample_size=sample['dxa_img'][0].shape) 
            transformed_mri_spatial_embeds = pca_transform(pca, mri_spatial_embeds, resample_size=sample['mri_img'][0].shape) 
            if MAKE_CORRELATION_MAPS: 
                make_correlation_map(dxa_spatial_embeds, mri_spatial_embeds, sample['dxa_img'], sample['mri_img'],idx) 

            if SAVE_IMAGES:
                plt.figure(figsize=(10,15)) 
                plt.subplot(221) 
                plt.axis('off') 
                plt.imshow(sample['dxa_img'][0], cmap='gray') 
                plt.subplot(222) 
                plt.axis('off') 
                plt.imshow(sample['mri_img'][0], cmap='gray') 
                plt.subplot(223) 
                plt.axis('off') 
                plt.imshow(transformed_dxa_spatial_embeds/transformed_dxa_spatial_embeds.max()) 
                plt.subplot(224) 
                plt.axis('off') 
                plt.imshow(transformed_mri_spatial_embeds/transformed_mri_spatial_embeds.max()) 
                pca_feature_images_output_path = os.path.join(OUTPUT_PATH, 'pca_spatial_feature_maps') 
                if not os.path.isdir(pca_feature_images_output_path): 
                    print(f"Making path to save files to at {pca_feature_images_output_path}") 
                    os.mkdir(pca_feature_images_output_path) 
                plt.savefig(os.path.join(pca_feature_images_output_path,f'example_saliency_map_{str(idx).zfill(3)}')) 
                plt.close('all') 
            if idx>NUM_PCA_FEATURE_MAPS_SAVED: 
                break 

@ex.capture 
def make_correlation_map(dxa_sp_embds, mri_sp_embds, dxa_img, mri_img,
                         save_idx, OUTPUT_PATH, USE_SOFTMAX, SAVE_IMAGES, pt=None): 
    mri_img_c, mri_img_h, mri_img_w = mri_img.size() 
    dxa_img_c, dxa_img_h, dxa_img_w = dxa_img.size() 
    mri_sp_embd_b, mri_sp_embd_c, mri_sp_embd_h, mri_sp_embd_w = mri_sp_embds.size() 
    dxa_sp_embd_b, dxa_sp_embd_c, dxa_sp_embd_h, dxa_sp_embd_w = dxa_sp_embds.size() 
    if pt == None: 
        pt = (int(np.round(np.random.random()*0.9*(mri_img_w-1))), int(np.round(np.random.random()*0.9*(mri_img_h-1))))


    pt_sp_embds = (np.round(pt[0]*mri_sp_embd_w/mri_img_w).astype(int), 
                   np.round(pt[1]*mri_sp_embd_h/mri_img_h).astype(int))


    # get 4d correlation
    dxa_sp_embds = dxa_sp_embds.view(dxa_sp_embd_b, dxa_sp_embd_c, dxa_sp_embd_h*dxa_sp_embd_w).permute(0,2,1) #batch, sp_dims, chnl
    mri_sp_embds = mri_sp_embds.view(mri_sp_embd_b, mri_sp_embd_c, mri_sp_embd_h*mri_sp_embd_w) # batch, chnl, sp_dims

    corr4d = torch.bmm(dxa_sp_embds, mri_sp_embds).view(dxa_sp_embd_b, dxa_sp_embd_h, dxa_sp_embd_w, mri_sp_embd_h, mri_sp_embd_w)

    corres_corr_map = corr4d[:,:,:,pt_sp_embds[1], pt_sp_embds[0]]
    if USE_SOFTMAX:
        corres_corr_map = F.softmax(corres_corr_map.contiguous().view(dxa_sp_embd_b, dxa_sp_embd_h*dxa_sp_embd_w)).view(dxa_sp_embd_b, dxa_sp_embd_h, dxa_sp_embd_w)
    
    corres_corr_map = F.interpolate(corres_corr_map[None], (dxa_img_h, dxa_img_w))[0]
    
    if SAVE_IMAGES:
        correlation_map_output_path = os.path.join(OUTPUT_PATH, 'correlation_maps')
        if not os.path.isdir(correlation_map_output_path): os.mkdir(correlation_map_output_path)
        plt.figure(figsize=(15,8))
        plt.subplot(131)
        plt.imshow(mri_img[0],cmap='gray')
        plt.scatter(pt[0],pt[1],marker='x',c='r',s=100)
        plt.subplot(132)
        plt.imshow(dxa_img[0],cmap='gray')
        plt.subplot(133)
        plt.imshow(corres_corr_map[0],cmap='Reds')
        plt.colorbar()
        plt.savefig(os.path.join(correlation_map_output_path, f'example_correlation_map_{str(save_idx).zfill(3)}'))




@ex.automain
def main(USE_FOUR_MODES, LOAD_FROM_PATH, POOLING_STRATEGY, DATASET_ROOT, 
         MODEL_TYPE, NORMALIZE_SPATIAL_FEATURES, USE_CACHED_PCA,OUTPUT_PATH, 
         MEASURE_TEST_PERFORMANCE, SAVE_IMAGES):

    if not SAVE_IMAGES: print('Warning: Not Saving Images')
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # load models
    if USE_FOUR_MODES:
        test_ds  = BothMidCoronalDataset(set_type='test' ,all_root=DATASET_ROOT, augment=False)
        from train_contrastive_all_modes_negative_mining import get_dataloaders, get_embeds_statistics
        train_dl, val_dl, test_dl = get_dataloaders(10, 10, 10, 10, 10,10, DATASET_ROOT)
    else:
        test_ds  = MidCoronalDataset(set_type='test' ,all_root=DATASET_ROOT, augment=False)
        from train_contrastive import get_dataloaders, get_embeds_statistics
        train_dl, val_dl, test_dl = get_dataloaders(10, 10, 10, 10, 10,10, False,DATASET_ROOT)

    # load models and print recorded performance
    dxa_model, mri_model, val_stats, epoch_no = load_models(LOAD_FROM_PATH, 
                                                            POOLING_STRATEGY, 
                                                            USE_FOUR_MODES, 
                                                            MODEL_TYPE)
    dxa_model.cuda().eval(); mri_model.cuda().eval()
    print('Recorded Val Stats:')
    print(val_stats)

    # measure to check discriminative performance is matched
    if MEASURE_TEST_PERFORMANCE:
        test_stats = test_epoch(dxa_model, mri_model, test_dl)
        print('Measured Test Stats:')
        print(test_stats)

    assess_approximation(dxa_model, mri_model, test_ds,normalize=True)

    pca = fit_pca(dxa_model, mri_model, test_ds)

    save_spatial_feature_maps(dxa_model, mri_model, test_ds, pca)

    # dist_matrix = get_distances_matrix(pooled_dxas, pooled_mris)
    import pdb; pdb.set_trace()
