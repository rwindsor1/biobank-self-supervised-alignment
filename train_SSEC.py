'''
Train self-supervised embedding correlator
Rhydian Windsor 07/02/20
'''

import sys, os, glob
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
from sklearn.metrics import auc

from models.SSECEncoders import VGGEncoder, VGGEncoderPoolAll
from datasets.BothMidCoronalDataset import BothMidCoronalDataset, BothMidCoronalHardNegativesDataset
from gen_utils import balanced_l1w_loss, grayscale, red
from loss_functions import SSECLoss
from sacred import Experiment
from sacred.observers import MongoObserver
# to make sensible stop
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'

ex = Experiment('TrainSSEC')
ex.captured_out_filter = lambda captured_output: "output capturing turned off."
ex.observers.append(MongoObserver(url='login1.triton.cluster:27017', db_name='SelfSupervisedProject'))

@ex.config
def expt_config():
    ADAM_BETAS = (0.9,0.999)
    NUM_WORKERS = 20
    TRAIN_NUM_WORKERS = NUM_WORKERS
    VAL_NUM_WORKERS   = NUM_WORKERS
    TEST_NUM_WORKERS  = NUM_WORKERS
    # Scan Encoder Details
    ENCODER_TYPE = VGGEncoder
    EMBEDDING_SIZE = 128
    MARGIN=0.1
    SOFTMAX_TEMP = 0.005
    # Use four modes of scans
    EQUAL_RES=True
    SINGLE_DXA=0 # 0 = Both, 1 = Bone, 2=Tissue
    SINGLE_MRI=0 # 0 = Both, 1 = Fat, 2=Water
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
    if COPY_TO_TEMP:
        DATASET_ROOT = '/tmp/rhydian'
    else:
        DATASET_ROOT = '/scratch/shared/beegfs/rhydian/UKBiobank'

@ex.capture
def get_dataloaders(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE, 
                    TRAIN_NUM_WORKERS, VAL_NUM_WORKERS, TEST_NUM_WORKERS, 
                    DATASET_ROOT, TRAINING_AUGMENTATION, EQUAL_RES,SINGLE_DXA, SINGLE_MRI):
    train_ds = BothMidCoronalDataset(set_type='train', all_root=DATASET_ROOT, single_dxa=SINGLE_DXA, single_mri=SINGLE_MRI, return_segmentations=False,equal_res=EQUAL_RES, augment=TRAINING_AUGMENTATION)
    val_ds   = BothMidCoronalDataset(set_type='val'  , all_root=DATASET_ROOT, single_dxa=SINGLE_DXA, single_mri=SINGLE_MRI, return_segmentations=False,equal_res=EQUAL_RES, augment=False)
    test_ds  = BothMidCoronalDataset(set_type='test' , all_root=DATASET_ROOT, single_dxa=SINGLE_DXA, single_mri=SINGLE_MRI, return_segmentations=False,equal_res=EQUAL_RES, augment=False)

    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=VAL_BATCH_SIZE,   num_workers=VAL_NUM_WORKERS,   shuffle=False, drop_last=True)
    test_dl  = DataLoader(test_ds,  batch_size=TEST_BATCH_SIZE,  num_workers=TEST_NUM_WORKERS,  shuffle=False, drop_last=True)
    return train_dl, val_dl, test_dl

@ex.capture
def load_models(ENCODER_TYPE, EMBEDDING_SIZE,
                LOAD_FROM_PATH, USE_CUDA,SINGLE_DXA, SINGLE_MRI):
    num_dxa_input_modes=2; num_mri_input_modes=2
    if SINGLE_DXA>0: num_dxa_input_modes=1
    if SINGLE_MRI>0: num_mri_input_modes=1
    mri_model = ENCODER_TYPE(input_modes=num_mri_input_modes, embedding_size=EMBEDDING_SIZE)
    dxa_model = ENCODER_TYPE(input_modes=num_dxa_input_modes, embedding_size=EMBEDDING_SIZE)
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
    if USE_CUDA:
        dxa_model.cuda()
        mri_model.cuda()
    return dxa_model, mri_model, val_stats, epochs

@ex.capture
def val_epoch(dxa_model, mri_model, dl, USE_CUDA, return_similarities=False, allow_rotations=False):
    dxa_model.eval()
    mri_model.eval()
    all_mri_ses = []
    all_dxa_ses = []
    pbar = tqdm(dl)
    for idx, sample in enumerate(pbar):
        mri_img = sample['mri_img']
        dxa_img = sample['dxa_img']
        if USE_CUDA:
            mri_img = mri_img.cuda()
            dxa_img = dxa_img.cuda()

        with torch.no_grad():
            dxa_ses = dxa_model(dxa_img).cpu()
            mri_ses = mri_model(mri_img).cpu()
            mri_ses = F.normalize(mri_ses,dim=1)
            dxa_ses = F.normalize(dxa_ses,dim=1)
            all_mri_ses.append(mri_ses)
            all_dxa_ses.append(dxa_ses)

    all_mri_ses = torch.cat(all_mri_ses)
    num_scans,_,_,_ = all_mri_ses.size()
    all_dxa_ses = torch.cat(all_dxa_ses)

    mri_b, mri_c, mri_h, mri_w = all_mri_ses.size()
    if allow_rotations:
        similarities = -torch.ones((num_scans,num_scans)).cuda()
        for rotation_angle in np.linspace(-5,5,20):
            print(rotation_angle)
            if USE_CUDA:
                all_mri_ses = all_mri_ses.cuda()
                all_mri_ses = TF.rotate(all_mri_ses,rotation_angle)
                all_dxa_ses = all_dxa_ses.cuda()
            with torch.no_grad():
                torch.cuda.empty_cache()
                corrs = (F.conv2d(all_dxa_ses, all_mri_ses)/(mri_h*mri_w)).view(num_scans,num_scans,-1)
                temp_similarities,_ = torch.max(corrs,dim=-1)
                similarities[temp_similarities>similarities] = temp_similarities[temp_similarities > similarities]
    else:
        if USE_CUDA:
            all_mri_ses = all_mri_ses.cuda()
            all_dxa_ses = all_dxa_ses.cuda()
        corrs = (F.conv2d(all_dxa_ses, all_mri_ses)/(mri_h*mri_w)).view(num_scans,num_scans,-1)
        similarities,_ = torch.max(corrs,dim=-1)
    print('Getting Rank Statistics...')
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
def train_epoch(dxa_model, mri_model, dl, optimizer, SOFTMAX_TEMP, USE_CUDA):
    dxa_model.train()
    mri_model.train()
    pbar = tqdm(dl)
    criterion = SSECLoss(SOFTMAX_TEMP)
    epoch_losses = torch.Tensor()
    epoch_correct = torch.Tensor()
    epoch_matching_similarities = torch.Tensor()
    epoch_non_matching_similarities = torch.Tensor()
    for idx, sample in enumerate(pbar):
        optimizer.zero_grad()
        mri_img = sample['mri_img']
        dxa_img = sample['dxa_img']
        if USE_CUDA:
            mri_img = mri_img.cuda()
            dxa_img = dxa_img.cuda()

        dxa_ses = dxa_model(dxa_img)
        mri_ses = mri_model(mri_img)


        mri_ses = F.normalize(mri_ses,dim=1)
        dxa_ses = F.normalize(dxa_ses,dim=1)

        mri_b, mri_c, mri_h, mri_w = mri_ses.size()
        correlations = (F.conv2d(dxa_ses, mri_ses)/(mri_h*mri_w)).view(mri_b,mri_b,-1)
        similarities, _ = torch.max(correlations,dim=-1)
        loss = criterion(similarities)
        loss.backward()
        optimizer.step()
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
def get_optimizers(dxa_model, mri_model, LR, ADAM_BETAS):
    optim = Adam(list(dxa_model.parameters()) + list(mri_model.parameters()), lr=LR, betas=ADAM_BETAS)
    return optim

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
def test(NOTE, ALLOW_ROTATIONS):
    train_dl, val_dl, test_dl = get_dataloaders()
    name = NOTE
    if ALLOW_ROTATIONS: name += 'with_rotations'
    dxa_model, mri_model, val_stats, epochs = load_models()
    val_stats, similarities = val_epoch(dxa_model, mri_model, test_dl, return_similarities=True, allow_rotations=ALLOW_ROTATIONS)
    print(val_stats)
    make_roc_curve(similarities)
    with open(f'roc_curve_statistics/{name}.pkl','wb') as f:
        pickle.dump([val_stats,similarities],f)


@ex.automain
def main(COPY_TO_TEMP, _run):
    if COPY_TO_TEMP and not os.path.isdir('/tmp/rhydian'): os.system('bash copy_all_to_temp.sh')
    train_dl, val_dl, test_dl = get_dataloaders()
    dxa_model, mri_model, val_stats, epochs = load_models()
    optim = get_optimizers(dxa_model, mri_model)
    best_rank = val_stats['mean_rank']
    val_stats = val_epoch(dxa_model, mri_model, val_dl)

    while True:
        print(f'Epoch {epochs}:')
        print('Training Epoch...')
        train_stats = train_epoch(dxa_model,mri_model, train_dl,optim)
        for key in train_stats:
            _run.log_scalar(f'training.{key}', train_stats[key])
        print(train_stats)
        print('Validating Epoch...')
        val_stats = val_epoch(dxa_model, mri_model, val_dl)
        for key in val_stats:
            _run.log_scalar(f'validation.{key}', val_stats[key])
        print(val_stats)
        if val_stats['mean_rank'] < best_rank:
            print('Saving Model')
            save_models(dxa_model,mri_model,val_stats,epochs)
            best_rank = val_stats['mean_rank']

        epochs += 1

