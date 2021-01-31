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
from datasets.BothMidCoronalDataset import BothMidCoronalDataset, BothMidCoronalHardNegativesDataset
from gen_utils import balanced_l1w_loss, grayscale, red
from loss_functions import ContrastiveLoss
from sacred import Experiment
from sacred.observers import MongoObserver
# to make sensible stop
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'

ex = Experiment('TrainContrastiveNegativeMining')
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
ex.observers.append(MongoObserver(url='login1.triton.cluster:27017', db_name='SelfSupervisedProject'))

@ex.config
def expt_config():
    LR = 0.00001
    ADAM_BETAS = (0.9,0.999)
    NUM_WORKERS = 20
    TRAIN_NUM_WORKERS = NUM_WORKERS
    VAL_NUM_WORKERS   = NUM_WORKERS
    TEST_NUM_WORKERS  = NUM_WORKERS
    BATCH_SIZE = 20
    TRAIN_BATCH_SIZE = BATCH_SIZE
    VAL_BATCH_SIZE = BATCH_SIZE
    TEST_BATCH_SIZE = BATCH_SIZE
    TRAINING_ITERATIONS = 20000/BATCH_SIZE 
    MARGIN = 0.1
    VALIDATION_ITERATIONS = 100
    NOTE=''
    POOLING_STRATEGY = 'max'
    assert POOLING_STRATEGY in ['max', 'average']
    BLANKING_AUGMENTATION = False
    USE_INSTANCE_LOSS = True
    BOTH_MODELS_WEIGHTS_PATH = './model_weights/ContrastiveModels' + NOTE
    if not USE_INSTANCE_LOSS:
        BOTH_MODELS_WEIGHTS_PATH += '_NoInstance'
    if BLANKING_AUGMENTATION:
        BOTH_MODELS_WEIGHTS_PATH += '_Blanking'
    if POOLING_STRATEGY == 'max':
        BOTH_MODELS_WEIGHTS_PATH += '_MaxPool'
    LOAD_FROM_PATH = BOTH_MODELS_WEIGHTS_PATH
    BOTH_MODELS_WEIGHTS_PATH += '_TripletLoss'
    SAVE_IMAGES_PATH = 'images/contrastive_examples' # the path to save responses of the network from the save_examples command
    SAVE_TEST_EMBEDDINGS_PATH = "saved_test_embeddings/contrastive_test_embeddings_" + BOTH_MODELS_WEIGHTS_PATH.split('/')[-1] + ".pkl"
    NUM_SAVE_IMAGES = 5 # how many images to save when running `save_examples`
    DATASET_ROOT = '/tmp/rhydian'

@ex.capture
def load_models(LOAD_FROM_PATH, POOLING_STRATEGY, use_cuda=True):
    dxa_model = SpatialVGGM(pooling=POOLING_STRATEGY, input_modes=2)
    mri_model = SpatialVGGM(pooling=POOLING_STRATEGY, input_modes=2)
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
                val_stats = {'mean_rank':999999999999}
                epochs = 0
    else:
        raise Exception(f'Could not find directory at {LOAD_FROM_PATH}')
        
    if use_cuda:
        dxa_model.cuda()
        mri_model.cuda()
    return dxa_model, mri_model, val_stats, epochs

@ex.capture
def get_optimizers(dxa_model, mri_model, LR, ADAM_BETAS):
    optim = Adam(list(dxa_model.parameters()) + list(mri_model.parameters()), lr=LR, betas=ADAM_BETAS)
    return optim

@ex.capture
def get_dataloaders(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE, TRAIN_NUM_WORKERS, VAL_NUM_WORKERS, TEST_NUM_WORKERS, DATASET_ROOT):
    train_ds = BothMidCoronalDataset(set_type='train', all_root=DATASET_ROOT, augment=True)
    val_ds   = BothMidCoronalDataset(set_type='val'  , all_root=DATASET_ROOT, augment=False)
    test_ds  = BothMidCoronalDataset(set_type='test' , all_root=DATASET_ROOT, augment=False)
    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=False, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=VAL_BATCH_SIZE,   num_workers=VAL_NUM_WORKERS,   shuffle=False, drop_last=True)
    test_dl  = DataLoader(test_ds,  batch_size=TEST_BATCH_SIZE,  num_workers=TEST_NUM_WORKERS,  shuffle=False, drop_last=True)
    return train_dl, val_dl, test_dl

@ex.capture
def get_embeddings(dxa_model, mri_model, dl, use_cached=True):
    dxa_model.eval()
    mri_model.eval()
    dxa_embeddings = []
    mri_embeddings = []
    cached_embeddings_path = 'temp/cached_embeddings.pkl'
    if use_cached and os.path.exists(cached_embeddings_path):
        with open(cached_embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(dl)):
                dxa_vols = sample['dxa_img']
                mri_vols = sample['mri_img']
                batch_dxa_embeds = dxa_model(dxa_vols.cuda()).squeeze(-1).squeeze(-1)
                batch_mri_embeds = mri_model(mri_vols.cuda()).squeeze(-1).squeeze(-1)
                batch_dxa_embeds = batch_dxa_embeds/batch_dxa_embeds.norm(dim=-1).unsqueeze(1)
                batch_mri_embeds = batch_mri_embeds/batch_mri_embeds.norm(dim=-1).unsqueeze(1)
                dxa_embeddings.append(batch_dxa_embeds.cpu().numpy())
                mri_embeddings.append(batch_mri_embeds.cpu().numpy())
        
        dxa_embeddings = np.concatenate(dxa_embeddings, axis=0)
        mri_embeddings = np.concatenate(mri_embeddings, axis=0)

        embeddings = {'mri_embeddings': mri_embeddings, 'dxa_embeddings': dxa_embeddings}
        with open(cached_embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
    return embeddings

@ex.capture
def val_epoch(dxa_model, mri_model,dl, BATCH_SIZE, MARGIN, USE_INSTANCE_LOSS, _run):
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

    _run.log_scalar('validation.epoch_loss',  np.mean(val_losses))
    _run.log_scalar('validation.mean_rank',   val_stats['mean_rank'])
    _run.log_scalar('validation.median_rank', val_stats['median_rank'])
    _run.log_scalar('validation.top_10',      val_stats['top_10'])
    _run.log_scalar('validation.top_5',       val_stats['top_5'])
    _run.log_scalar('validation.top_1',       val_stats['top_1'])
    return val_stats

@ex.capture
def get_hard_negatives(embeddings, MARGIN, use_cached=True):
    dxa_embeddings = embeddings['dxa_embeddings']
    mri_embeddings = embeddings['mri_embeddings']
    all_triplets = []
    num_hard_negatives = 0
    num_semi_hard_negatives = 0
    cached_hard_negatives_path = './temp/hard_negatives.pkl'
    if use_cached and os.path.exists(cached_hard_negatives_path):
        with open(cached_hard_negatives_path, 'rb') as f:
            all_triplets = pickle.load(f)
    else:
        matching_pair_distances = np.linalg.norm(dxa_embeddings - mri_embeddings, axis=1)
        for dxa_idx in tqdm(range(dxa_embeddings.shape[0])):
            non_matching_pair_distances = np.linalg.norm((dxa_embeddings[None,dxa_idx,:] - mri_embeddings[:,:]),axis=1)
            seperations = matching_pair_distances[dxa_idx] - non_matching_pair_distances
            hard_negatives = np.where(seperations > 0)[0].tolist()
            semi_hard_negatives = np.where((seperations > -MARGIN)*(seperations<0))[0].tolist()
            num_hard_negatives += len(hard_negatives)
            num_semi_hard_negatives += len(semi_hard_negatives) - 1
            candidate_triplets = np.concatenate([semi_hard_negatives, hard_negatives])
            all_triplets += [(dxa_idx, mri_idx) for mri_idx in candidate_triplets if dxa_idx != mri_idx]
    return all_triplets, num_hard_negatives, num_semi_hard_negatives

@ex.capture
def get_embeds_statistics(dxa_embeds : torch.Tensor, mri_embeds : torch.Tensor):
        norm_dxa_embeds = dxa_embeds/dxa_embeds.norm(dim=-1).unsqueeze(1)
        norm_mri_embeds = mri_embeds/mri_embeds.norm(dim=-1).unsqueeze(1)

        similarities_matrix = norm_dxa_embeds@norm_mri_embeds.T
        sorted_similarities_values, sorted_similarities_idxs = similarities_matrix.sort(dim=1,descending=True)
        ranks = []
        for idx, row in enumerate(sorted_similarities_idxs):
            rank = np.where(row.numpy()==idx)[0][0]
            ranks.append(rank)
        ranks = np.array(ranks)
        mean_rank = np.mean(ranks)
        median_rank = np.median(ranks)
        top_10 = np.sum(ranks<10) / len(ranks)
        top_5  = np.sum(ranks<5)  / len(ranks)
        top_1  = np.sum(ranks<1)  / len(ranks)
        return {'mean_rank': mean_rank, 'median_rank': median_rank, 'top_10': top_10, 'top_5': top_5, 'top_1': top_1}

@ex.capture
def train_epoch(train_hard_negatives, dxa_model, mri_model, optim, TRAIN_BATCH_SIZE, TRAIN_NUM_WORKERS, TRAINING_ITERATIONS, DATASET_ROOT, MARGIN):
    ds = BothMidCoronalHardNegativesDataset(train_hard_negatives, all_root=DATASET_ROOT)
    dl = DataLoader(ds, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=True, drop_last=True)
    triplet_loss = nn.TripletMarginLoss(margin=MARGIN)
    dxa_model.train()
    mri_model.train()
    pbar = tqdm(dl)
    for idx, sample in enumerate(pbar):            
        optim.zero_grad() 
        dxa_vols1 = sample['dxa_img1']
        mri_vols1 = sample['mri_img1']
        dxa_vols2 = sample['dxa_img2']
        mri_vols2 = sample['mri_img2']
        dxa_embeds1 = F.normalize(dxa_model(dxa_vols1.cuda()).squeeze(-1).squeeze(-1), dim=-1)
        mri_embeds1 = F.normalize(mri_model(mri_vols1.cuda()).squeeze(-1).squeeze(-1), dim=-1)
        dxa_embeds2 = F.normalize(dxa_model(dxa_vols2.cuda()).squeeze(-1).squeeze(-1), dim=-1)
        mri_embeds2 = F.normalize(mri_model(mri_vols2.cuda()).squeeze(-1).squeeze(-1), dim=-1)
        dxa_triplet_loss = triplet_loss(dxa_embeds1, mri_embeds1, mri_embeds2)
        mri_triplet_loss = triplet_loss(mri_embeds1, dxa_embeds1, dxa_embeds2)
        total_loss = dxa_triplet_loss + mri_triplet_loss
        total_loss.backward()
        optim.step()
        if idx >= TRAINING_ITERATIONS: break


@ex.capture
def save_models(dxa_model, mri_model, val_stats, epochs, BOTH_MODELS_WEIGHTS_PATH):
    print(f'==> Saving Model Weights to {BOTH_MODELS_WEIGHTS_PATH}')
    state = {'dxa_model_weights': dxa_model.state_dict(),
             'mri_model_weights': mri_model.state_dict(),
             'val_stats'        : val_stats,
             'epochs'           : epochs
            }
    if not os.path.isdir(BOTH_MODELS_WEIGHTS_PATH):
        os.mkdir(BOTH_MODELS_WEIGHTS_PATH)
    previous_checkpoints = glob.glob(BOTH_MODELS_WEIGHTS_PATH + '/ckpt*.pt', recursive=True)
    for previous_checkpoint in previous_checkpoints:
       os.remove(previous_checkpoint)
    torch.save(state, BOTH_MODELS_WEIGHTS_PATH + '/ckpt' + str(epochs) + '.pt')
    return

@ex.automain
def main(_run):
    dxa_model, mri_model, val_stats, epoch_no = load_models()
    epochs = 1
    train_dl, val_dl, test_dl = get_dataloaders()
    optim  = get_optimizers(dxa_model, mri_model)
    val_stats = val_epoch(dxa_model, mri_model, val_dl)
    best_mean_rank = val_stats['mean_rank']
    print(val_stats)
    while True:
        print(f"Epoch {epochs}, calculating embeddings")
        train_embeddings = get_embeddings(dxa_model, mri_model, train_dl, use_cached=False)
        print(f"Epoch {epochs}, finding hard negatives")
        train_hard_negatives, num_hard_negatives, num_semi_hard_negatives = get_hard_negatives(train_embeddings, use_cached=False)
        _run.log_scalar('training.hard_negatives',  num_hard_negatives)
        _run.log_scalar('training.semi_hard_negatives',  num_semi_hard_negatives)
        print(f"Epoch {epochs}, training")
        train_epoch(train_hard_negatives, dxa_model, mri_model, optim)
        print(f"Epoch {epochs}, validating")
        val_stats =  val_epoch(dxa_model, mri_model, val_dl)
        print(val_stats)
        if val_stats['mean_rank'] < best_mean_rank:
            save_models(dxa_model, mri_model, val_stats, epochs)

        epochs += 1

        # main loop
