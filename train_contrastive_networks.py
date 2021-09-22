'''
Train self-supervised embedding correlator
Rhydian Windsor 07/02/20
'''

import glob
import os
import pickle

from tqdm.utils import SimpleTextIOWrapper
from src.loss_fns import NCELoss

from torch import optim
from src.utils.misc import load_checkpoint, optimiser_to

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# from gen_utils import balanced_l1w_loss, grayscale, red
# from loss_functions import SSECLoss

from sacred import SETTINGS, Experiment
from sacred.observers import MongoObserver
from sklearn.metrics import auc
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.datasets import CoronalScanPairsDataset
from src.models import VGGEncoder, VGGEncoderPoolAll
from src.utils import load_checkpoint, save_checkpoint, get_batch_corrrelations, get_dataset_similarities

SETTINGS['CAPTURE_MODE'] = 'sys'

ex = Experiment('TrainScanEncoders')
ex.captured_out_filter = lambda captured_output: "output capturing turned off."
ex.observers.append(MongoObserver(url='login1.triton.cluster:27017'))

@ex.config
def expt_config():
    ADAM_BETAS = (0.9,0.999)
    NUM_WORKERS = 20
    TRAIN_NUM_WORKERS = NUM_WORKERS
    VAL_NUM_WORKERS   = NUM_WORKERS
    TEST_NUM_WORKERS  = NUM_WORKERS
    # Scan Encoder Details
    ENCODER_TYPE = 'VGGEncoder' # 'VGGEncoder' (ours), 'VGGEncoderPoolAll' (baseline) 
    EMBEDDING_SIZE = 128
    MARGIN=0.1
    SOFTMAX_TEMP = 0.005
    # Use four modes of scans
    MRI_SEQS=['fat_scan','water_scan'] # 0 = Both, 1 = Bone, 2=Tissue
    DXA_SEQS=['bone','tissue'] # 0 = Both, 1 = Fat, 2=Water
    BATCH_SIZE=10
    TRAIN_BATCH_SIZE = BATCH_SIZE
    VAL_BATCH_SIZE = BATCH_SIZE
    TEST_BATCH_SIZE = BATCH_SIZE
    USE_CUDA=True
    TRAINING_ITERATIONS = 20000/BATCH_SIZE
    MARGIN = 0.1
    VALIDATION_ITERATIONS = 100
    TRAINING_AUGMENTATION = True
    ALLOW_ROTATIONS=False
    POOL_SPATIAL_MAPS=False # for baseline
    NOTE=''
    LR=0.00001
    COPY_TO_TEMP=False
    BOTH_MODELS_WEIGHTS_PATH = './model_weights/SSECEncoders' + NOTE
    LOAD_FROM_PATH = BOTH_MODELS_WEIGHTS_PATH
    SAVE_IMAGES_PATH = 'images/contrastive_examples' # the path to save responses of the network from the save_examples command
    SAVE_ROC_PATH = 'images/roc_curve.png'
    DATASET_ROOT = '/work/rhydian/UKBB_Downloads'
    TMP_DIR = '/tmp/rhydian/UKBB_Downloads'

@ex.capture
def get_dataloaders(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE, 
                    TRAIN_NUM_WORKERS, VAL_NUM_WORKERS, TEST_NUM_WORKERS, 
                    DATASET_ROOT, COPY_TO_TEMP, TMP_DIR,
                    TRAINING_AUGMENTATION, MRI_SEQS, DXA_SEQS):
    if COPY_TO_TEMP: DATASET_ROOT=TMP_DIR
    print(DATASET_ROOT)
    train_ds = CoronalScanPairsDataset(set_type='train', root=DATASET_ROOT, mri_seqs=MRI_SEQS, dxa_seqs=DXA_SEQS, augment=TRAINING_AUGMENTATION)
    val_ds   = CoronalScanPairsDataset(set_type='val'  , root=DATASET_ROOT, mri_seqs=MRI_SEQS, dxa_seqs=DXA_SEQS, augment=False)
    test_ds  = CoronalScanPairsDataset(set_type='test' , root=DATASET_ROOT, mri_seqs=MRI_SEQS, dxa_seqs=DXA_SEQS, augment=False)

    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=VAL_BATCH_SIZE,   num_workers=VAL_NUM_WORKERS,   shuffle=False, drop_last=True)
    test_dl  = DataLoader(test_ds,  batch_size=TEST_BATCH_SIZE,  num_workers=TEST_NUM_WORKERS,  shuffle=False, drop_last=True)
    return train_dl, val_dl, test_dl



@ex.capture
def load_model_and_optimisers(ENCODER_TYPE, EMBEDDING_SIZE,
                              LOAD_FROM_PATH, USE_CUDA, 
                              MRI_SEQS, DXA_SEQS, LR, ADAM_BETAS):

    mri_model = eval(ENCODER_TYPE)(input_modes=len(DXA_SEQS), embedding_size=EMBEDDING_SIZE)
    dxa_model = eval(ENCODER_TYPE)(input_modes=len(MRI_SEQS), embedding_size=EMBEDDING_SIZE)
    optimiser = Adam(list(dxa_model.parameters()) + list(mri_model.parameters()), lr=LR, betas=ADAM_BETAS)
    print(f'Trying to load model from {LOAD_FROM_PATH}')
    dxa_model, mri_model, optimiser,val_stats,epochs  = load_checkpoint(dxa_model, mri_model, optimiser, 
                                                                            LOAD_FROM_PATH, USE_CUDA)
    dxa_model = nn.DataParallel(dxa_model)
    mri_model = nn.DataParallel(mri_model)
    return dxa_model, mri_model, optimiser, val_stats, epochs

@ex.capture
def validate(dxa_model, mri_model, dl, USE_CUDA, return_similarities=False):
    dxa_model.eval()
    mri_model.eval()
    all_mri_ses = []
    all_dxa_ses = []
    pbar = tqdm(dl)
    # begin by encoding all scans
    print('Encoding scans')
    for idx, sample in enumerate(pbar):
        mri_img = sample['mri_img']
        dxa_img = sample['dxa_img']
        if USE_CUDA:
            mri_img = mri_img.cuda()
            dxa_img = dxa_img.cuda()

        with torch.no_grad():
            dxa_ses = dxa_model(dxa_img).cpu()
            mri_ses = mri_model(mri_img).cpu()
            all_mri_ses.append(mri_ses)
            all_dxa_ses.append(dxa_ses)

    all_mri_ses = torch.cat(all_mri_ses)
    num_scans,_,_,_ = all_mri_ses.size()
    all_dxa_ses = torch.cat(all_dxa_ses)

    # now correlate encodings
    mri_b, mri_c, mri_h, mri_w = all_mri_ses.size()
    if USE_CUDA:
        all_mri_ses = all_mri_ses.cuda()
        all_dxa_ses = all_dxa_ses.cuda()

    print('Calculating encoding similarities + statistics')
    similarities = get_dataset_similarities(all_dxa_ses, all_mri_ses)
    # corrs = (F.conv2d(all_dxa_ses, all_mri_ses)/(mri_h*mri_w)).view(num_scans,num_scans,-1)
    rank_stats = get_rank_statistics(similarities)
    if not return_similarities:
        return rank_stats
    else:
        return rank_stats, similarities


def get_rank_statistics(similarities_matrix):
        sorted_similarities_values, sorted_similarities_idxs = similarities_matrix.sort(dim=1,descending=True)
        ranks = []
        for idx, row in enumerate(tqdm(sorted_similarities_idxs)):
            rank = torch.where(row==idx)[0][0]
            ranks.append(rank.cpu())
        ranks = np.array(ranks)
        mean_rank = np.mean(ranks)
        median_rank = np.median(ranks)
        top_10 = np.sum(ranks<10) / len(ranks)
        top_5  = np.sum(ranks<5)  / len(ranks)
        top_1  = np.sum(ranks<1)  / len(ranks)

        ranks_stats = {'mean_rank': mean_rank, 'median_rank': median_rank,
                       'top_10': top_10, 'top_5': top_5, 'top_1':top_1}

        return ranks_stats

@ex.capture
def run_epoch(dxa_model, mri_model, dl, optimiser, SOFTMAX_TEMP, USE_CUDA, train=False):
    if train: dxa_model.train(); mri_model.train()
    else: dxa_model.eval(); mri_model.eval()
    pbar = tqdm(dl)
    criterion = NCELoss(SOFTMAX_TEMP)
    epoch_losses = torch.Tensor()
    epoch_correct = torch.Tensor()
    epoch_matching_similarities = torch.Tensor()
    epoch_non_matching_similarities = torch.Tensor()
    for idx, sample in enumerate(pbar):
        with torch.set_grad_enabled(train):
            if train: optimiser.zero_grad()
            mri_img = sample['mri_img']
            dxa_img = sample['dxa_img']
            if USE_CUDA:
                mri_img = mri_img.cuda()
                dxa_img = dxa_img.cuda()

            dxa_ses = dxa_model(dxa_img)
            mri_ses = mri_model(mri_img)

            # correlate and measure similarities
            correlations = get_batch_corrrelations(dxa_ses, mri_ses)
            similarities, _ = torch.max(correlations.flatten(start_dim=2),dim=-1)

            loss = criterion(similarities)
            if train:
                loss.backward()
                optimiser.step()

            correct = (similarities.argmax(dim=1) == torch.arange(similarities.shape[0]).cuda()).cpu()
            matching_similarities = similarities.diag().cpu()
            non_matching_similarities = similarities[~torch.eye(similarities.shape[0]).bool()].view(-1).cpu()
            # take sub sample of non matching similarities
            non_matching_similarities = non_matching_similarities[torch.randperm(non_matching_similarities.shape[0])][:100]

            epoch_matching_similarities = torch.cat([epoch_matching_similarities, matching_similarities])
            epoch_non_matching_similarities = torch.cat([epoch_non_matching_similarities, non_matching_similarities])
            epoch_correct = torch.cat([epoch_correct,correct.float()])
            epoch_losses = torch.cat([epoch_losses, loss[None].cpu()])
            pbar.set_description(f"Loss:{epoch_losses[-100:].mean():.4} Correct: {epoch_correct[-100:].mean():.4}")

    mean_loss =  epoch_losses.mean()
    mean_correct = epoch_correct.mean()
    mean_matching_similarity = matching_similarities.mean()
    mean_non_matching_similarity = non_matching_similarities.mean()
    train_stats = {'mean_loss':mean_loss.item(),'mean_correct':mean_correct.item(),
                   'mean_matching':mean_matching_similarity.item(),
                   'mean_non_matching':mean_non_matching_similarity.item()}
    return train_stats



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


@ex.capture
def make_roc_curve(similarities, SAVE_ROC_PATH):
    roc_points = []
    for threshold in np.linspace(0,1,1000):
        tpr = (similarities.diag()>threshold).sum()/(similarities.diag().shape[0])
        fpr = (similarities[~np.eye(similarities.shape[0],dtype=bool)]>threshold).sum()/similarities[~np.eye(similarities.shape[0],dtype=bool)].shape[0]
        roc_points.append([fpr.item(), tpr.item()])

    roc_points.sort(key=lambda x:x[0])
    roc_points = np.array(roc_points)
    plt.figure(figsize=(10,10))
    plt.plot(roc_points[:,0], roc_points[:,1], linewidth=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    epsilon=0.01
    plt.xlim([0-epsilon,1+epsilon])
    plt.ylim([0-epsilon,1+epsilon])
    plt.plot([0,1],[0,1],color='gray', linestyle='--')
    plt.savefig(SAVE_ROC_PATH)
    print(f"AUC: {auc(roc_points[:,0], roc_points[:,1])}")


@ex.command(unobserved=True)
def setup(_run):
    print('Loading Dataloaders...')
    train_dl, val_dl, test_dl = get_dataloaders()
    print('Loading Models and Optimizers...')
    dxa_model, mri_model, optimiser, val_stats, epochs = load_model_and_optimisers()
    return train_dl, val_dl, test_dl, dxa_model, mri_model, optimiser, val_stats, epochs

@ex.command(unobserved=True)
def test(NOTE,USE_CUDA):
    name=NOTE
    if USE_CUDA: os.system('nvidia-smi')
    train_dl, val_dl, test_dl, dxa_model, mri_model, optimiser, val_stats, epochs = setup()
    val_stats, similarities = validate(dxa_model, mri_model, test_dl, return_similarities=True)
    print(val_stats)
    make_roc_curve(similarities)
    with open(f'roc_curve_statistics/{name}.pkl','wb') as f:
        pickle.dump([val_stats,similarities],f)

@ex.automain
def main(COPY_TO_TEMP, DATASET_ROOT, TMP_DIR, USE_CUDA,BOTH_MODELS_WEIGHTS_PATH, _run):
    if COPY_TO_TEMP and not(os.path.isdir(TMP_DIR)): print('Copying to temp directory');os.system(f'bash copy_to_temp.sh {DATASET_ROOT} {TMP_DIR}')
    if USE_CUDA: os.system('nvidia-smi')
    train_dl, val_dl, test_dl, dxa_model, mri_model, optimiser, val_stats, epochs = setup()
    best_rank = val_stats['mean_rank']
    val_stats = validate(dxa_model, mri_model, val_dl)

    while True:
        print(f'Epoch {epochs}:\nTraining Epoch...')
        train_stats = run_epoch(dxa_model,mri_model, train_dl,optimiser, train=True)
        for key in train_stats:
            _run.log_scalar(f'training.{key}', train_stats[key])
        print(train_stats)
        print(f'Epoch {epochs}:Validating Epoch...')
        val_stats = validate(dxa_model, mri_model, val_dl)
        for key in val_stats:
            _run.log_scalar(f'validation.{key}', val_stats[key])
        print(val_stats)
        if val_stats['mean_rank'] < best_rank:
            print('Saving Model')
            save_checkpoint(dxa_model,mri_model,optimiser,val_stats,epochs,BOTH_MODELS_WEIGHTS_PATH)
            best_rank = val_stats['mean_rank']

        epochs += 1

