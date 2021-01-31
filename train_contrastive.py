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
from datasets.MidCoronalDataset import MidCoronalDataset
from gen_utils import balanced_l1w_loss, grayscale, red
from loss_functions import ContrastiveLoss
from sacred import Experiment
from sacred.observers import MongoObserver
# to make sensible stop
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'

ex = Experiment('TrainContrastive2')
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
ex.observers.append(MongoObserver(url='login1.triton.cluster:27017', db_name='SelfSupervisedProject'))

@ex.config
def expt_config():
    LR = 0.000001
    ADAM_BETAS = (0.9,0.999)
    NUM_WORKERS = 20
    TRAIN_NUM_WORKERS = NUM_WORKERS
    VAL_NUM_WORKERS   = NUM_WORKERS
    TEST_NUM_WORKERS  = NUM_WORKERS
    BATCH_SIZE = 20
    TRAIN_BATCH_SIZE = BATCH_SIZE
    VAL_BATCH_SIZE = BATCH_SIZE
    TEST_BATCH_SIZE = BATCH_SIZE
    TRAINING_ITERATIONS = 100
    MARGIN = 1
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
    SAVE_IMAGES_PATH = 'images/contrastive_examples' # the path to save responses of the network from the save_examples command
    SAVE_TEST_EMBEDDINGS_PATH = "saved_test_embeddings/contrastive_test_embeddings_" + BOTH_MODELS_WEIGHTS_PATH.split('/')[-1] + ".pkl"
    NUM_SAVE_IMAGES = 5 # how many images to save when running `save_examples`
    DATASET_ROOT = '/tmp/rhydian'

@ex.capture
def get_dataloaders(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE, TRAIN_NUM_WORKERS, VAL_NUM_WORKERS, TEST_NUM_WORKERS, BLANKING_AUGMENTATION, DATASET_ROOT):
    train_ds = MidCoronalDataset(set_type='train',all_root=DATASET_ROOT,     augment=True, blanking_augment=BLANKING_AUGMENTATION)
    val_ds   = MidCoronalDataset(set_type='val'  ,all_root=DATASET_ROOT,       augment=False)
    test_ds  = MidCoronalDataset(set_type='test' ,all_root=DATASET_ROOT,      augment=False)
    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=VAL_BATCH_SIZE, num_workers=VAL_NUM_WORKERS,     shuffle=False,      drop_last=True)
    test_dl  = DataLoader(test_ds,  batch_size=TEST_BATCH_SIZE, num_workers=TEST_NUM_WORKERS,   shuffle=False,   drop_last=True)
    return train_dl, val_dl, test_dl

@ex.capture
def load_models(LOAD_FROM_PATH, POOLING_STRATEGY):
    # TODO: Implement model loading/saving capacity
    dxa_model = SpatialVGGM(pooling=POOLING_STRATEGY)
    mri_model = SpatialVGGM(pooling=POOLING_STRATEGY)
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
                best_loss = checkpoint['loss']
                epochs = checkpoint['epochs'] 
                print(f"==> Resuming model trained for {epochs} epochs...")
            else:
                print("==> Training Fresh Model ")
                best_loss = 99999999999999
                epochs = 0
    else:
        raise Exception(f'Could not find directory at {LOAD_FROM_PATH}')

    return dxa_model, mri_model, best_loss, epochs


@ex.capture
def save_models(dxa_model, mri_model, val_loss, epochs, BOTH_MODELS_WEIGHTS_PATH):
    print('==> Saving Model Weights')
    state = {'dxa_model_weights': dxa_model.state_dict(),
             'mri_model_weights': mri_model.state_dict(),
             'loss'             : val_loss,
             'epochs'           : epochs
            }
    previous_checkpoints = glob.glob(BOTH_MODELS_WEIGHTS_PATH + '/ckpt*.pt', recursive=True)
    for previous_checkpoint in previous_checkpoints:
       os.remove(previous_checkpoint)
    torch.save(state, BOTH_MODELS_WEIGHTS_PATH + '/ckpt' + str(epochs) + '.pt')
    return

@ex.capture
def get_optimizers(dxa_model, mri_model, LR, ADAM_BETAS):
    optim = Adam(list(dxa_model.parameters()) + list(mri_model.parameters()), lr=LR, betas=ADAM_BETAS)
    return optim

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
def train_epoch(dxa_model, mri_model, dl, optim, BATCH_SIZE, MARGIN,USE_INSTANCE_LOSS, _run):
    train_pbar = tqdm(dl)
    # train
    dxa_embeds = []
    mri_embeds = []
    train_losses = []
    criterion = ContrastiveLoss(BATCH_SIZE,MARGIN)
    dxa_model.train()
    mri_model.train()
    for idx, sample in enumerate(train_pbar):
        optim.zero_grad()
        dxa_vols = sample['dxa_img']
        mri_vols = sample['mri_img']
        dxa_embed = dxa_model(dxa_vols.cuda())
        mri_embed = mri_model(mri_vols.cuda())
        dxa_embed = F.normalize(dxa_embed.squeeze(-1).squeeze(-1),dim=-1)
        mri_embed = F.normalize(mri_embed.squeeze(-1).squeeze(-1),dim=-1)
        if USE_INSTANCE_LOSS:
            loss = criterion(dxa_embed, mri_embed) + criterion(dxa_embed, dxa_embed) + criterion(mri_embed, mri_embed)
        else:
            loss = criterion(dxa_embed, mri_embed)
        loss.backward()
        train_losses.append(loss.item())
        optim.step()
        dxa_embeds.append(dxa_embed.cpu())
        mri_embeds.append(mri_embed.cpu())
        train_pbar.set_description(f"{loss.item():.3}")

    dxa_embeds = torch.cat(dxa_embeds, dim=0)
    mri_embeds = torch.cat(mri_embeds, dim=0)
    train_stats = get_embeds_statistics(dxa_embeds, mri_embeds)
    _run.log_scalar('training.epoch_loss',  np.mean(train_losses))
    _run.log_scalar('training.mean_rank',   train_stats['mean_rank'])
    _run.log_scalar('training.median_rank', train_stats['median_rank'])
    _run.log_scalar('training.top_10',      train_stats['top_10'])
    _run.log_scalar('training.top_5',       train_stats['top_5'])
    _run.log_scalar('training.top_1',       train_stats['top_1'])
    return np.mean(train_losses)

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
    return np.mean(val_losses)

@ex.command(unobserved=True)
def test(BATCH_SIZE, MARGIN, SAVE_TEST_EMBEDDINGS_PATH, USE_INSTANCE_LOSS):
    criterion = ContrastiveLoss(BATCH_SIZE, MARGIN)
    train_dl, val_dl, test_dl = get_dataloaders()
    dxa_model, mri_model, best_loss, epoch_no = load_models()
    dxa_model.cuda().eval()
    mri_model.cuda().eval()
    pbar = tqdm(test_dl)
    dxa_embeds, mri_embeds, test_losses = [],[],[]
    print("Testing:")
    for idx, sample in enumerate(pbar):
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
        test_losses.append(loss.item())
        dxa_embeds.append(dxa_embed.cpu())
        mri_embeds.append(mri_embed.cpu())
        pbar.set_description(f"{loss.item():.3}")
    dxa_embeds = torch.cat(dxa_embeds, dim=0).cpu()
    mri_embeds = torch.cat(mri_embeds, dim=0).cpu()
    test_stats = get_embeds_statistics(dxa_embeds, mri_embeds)
    results_dict = {'dxa_embeds': dxa_embeds,
                    'mri_embeds': mri_embeds,
                    'test_stats':  test_stats}
    print("Results:")
    print(test_stats)
    print(f"Saving resulting embeddings to {SAVE_TEST_EMBEDDINGS_PATH}")
    with open(SAVE_TEST_EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(results_dict,f)
    print("Run make_contrastive_plots.py to make ROC/error curves")


@ex.automain
def main(BATCH_SIZE, _run,MARGIN):
    train_dl, val_dl, test_dl = get_dataloaders()
    dxa_model, mri_model, best_loss, epoch_no = load_models()
    dxa_model.cuda()
    mri_model.cuda()
    optim  = get_optimizers(dxa_model, mri_model)
    while True:
        print(f"Epoch {epoch_no}:")
        print("Training...")
        train_loss = train_epoch(dxa_model, mri_model,train_dl, optim)
        print("Validating...")
        val_loss = val_epoch(dxa_model, mri_model,val_dl)

        if val_loss  < best_loss:
            save_models(dxa_model, mri_model, val_loss, epoch_no)
            best_loss = val_loss

        epoch_no += 1
