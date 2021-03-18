import sys, os, glob
sys.path.append('/users/rhydian/self-supervised-project')
from gen_utils import *
import torch
from datasets.BothMidCoronalDataset import BothMidCoronalRegistrationDataset
from models.SSECEncoders import VGGEncoder
from registration_model2 import vgg16_bn
from registration_model3 import RefinementRegressor
from torch.utils.data import DataLoader
from sacred import Experiment
from tqdm import tqdm
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt
from sacred.observers import MongoObserver
import torchvision.transforms.functional as TF
import train_SSEC

ex=Experiment('TrainAngleRegressor')
ex.captured_out_filter = lambda captured_output: "output capturing turned off."
ex.observers.append(MongoObserver(url='login1.triton.cluster:27017', db_name='SelfSupervisedProject'))

@ex.config
def config():
    BATCH_SIZE=10
    NUM_WORKERS=10
    ENCODER_LOAD_FROM_PATH='/users/rhydian/self-supervised-project/model_weights/SSECEncodersBothBoth'
    LOAD_FROM_PATH='/users/rhydian/self-supervised-project/model_weights/angle_regressor_maps4'
    USE_CUDA=True
    ADAM_BETAS = (0.9,0.999)
    LR=0.001
    SAVE_IMAGES=True
    ZERO_PARAMETER=1
    AUG_ROT_VAL=20
    AUG_TRANS_VAL=30
    USE_TEMP=False
    EMBEDDING_SIZE=128
    if USE_TEMP: DATASET_ROOT='/tmp/rhydian'
    else: DATASET_ROOT='/scratch/shared/beegfs/rhydian/UKBiobank'

@ex.capture
def get_dataloaders(BATCH_SIZE, NUM_WORKERS,AUG_ROT_VAL, AUG_TRANS_VAL, DATASET_ROOT):
    train_ds = BothMidCoronalRegistrationDataset(set_type='train', all_root=DATASET_ROOT, rand_angle=AUG_ROT_VAL, rand_translation=AUG_TRANS_VAL)
    val_ds =   BothMidCoronalRegistrationDataset(set_type='val',   all_root=DATASET_ROOT, rand_angle=AUG_ROT_VAL, rand_translation=AUG_TRANS_VAL)
    test_ds =  BothMidCoronalRegistrationDataset(set_type='test',  all_root=DATASET_ROOT, rand_angle=AUG_ROT_VAL, rand_translation=AUG_TRANS_VAL)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    return train_dl, val_dl, test_dl

@ex.capture
def load_models(ENCODER_LOAD_FROM_PATH,USE_CUDA, LOAD_FROM_PATH, EMBEDDING_SIZE):
    dxa_model, mri_model,_,_ = train_SSEC.load_models(VGGEncoder, 128, ENCODER_LOAD_FROM_PATH,USE_CUDA,False,False)
    for parameter in dxa_model.module.parameters():
        parameter.requires_grad=False
    for parameter in mri_model.module.parameters():
        parameter.requires_grad=False
    #model = vgg16_bn(num_classes=3)
    model = RefinementRegressor(EMBEDDING_SIZE)
    #model = nn.DataParallell(model)
    model_dict = model.state_dict()
    print(f'Trying to load from {LOAD_FROM_PATH}')
    if not os.path.isdir(LOAD_FROM_PATH):
        os.mkdir(LOAD_FROM_PATH)
    # load model weights
    if os.path.isdir(LOAD_FROM_PATH):
            list_of_pt = glob.glob(LOAD_FROM_PATH + '/*.pt')
            if len(list_of_pt):
                latest_pt = max(list_of_pt, key=os.path.getctime)
                checkpoint = torch.load(latest_pt, map_location=torch.device('cpu'))
                model_dict.update(checkpoint['model_weights'])
                model.load_state_dict(model_dict)
                best_loss = checkpoint['best_loss']
                epochs = checkpoint['epochs']
                print(f"==> Resuming model trained for {epochs} epochs...")
            else:
                print("==> Training Fresh Model ")
                best_loss = 99999999999999
                epochs = 0
    else:
        raise Exception(f'Could not find directory at {LOAD_FROM_PATH}')
    if USE_CUDA:
        model.cuda()
    return model, dxa_model, mri_model, best_loss, epochs


@ex.capture
def get_optim(model, LR, ADAM_BETAS):
    optim = Adam(model.parameters(), lr=LR, betas=ADAM_BETAS)
    return optim

def align_maps(x1,x2):
    responses = F.conv2d(x1, x2)
    max_responses = responses.flatten(start_dim=-2).argmax(dim=-1)
    new_x2 = torch.zeros_like(x1)
    max_args = torch.stack([max_responses.diag() // responses.shape[-1], max_responses.diag() % responses.shape[-1]],dim=1)
    diff = [x1.shape[-2]-x2.shape[-2], x1.shape[-1] - x2.shape[-1]]
    for idx in range(x1.shape[0]): new_x2[idx] = F.pad(x2[idx],(max_args[idx][1],diff[1]-max_args[idx][1],max_args[idx][0],diff[0]-max_args[idx][0]))
    xs_cat = torch.cat([x1,new_x2],dim=1)
    return xs_cat, max_args[:,0], max_args[:,1]

def cat_ses(mri_ses, dxa_ses):
    diff = [dxa_ses.shape[-2]-mri_ses.shape[-2], dxa_ses.shape[-1] - mri_ses.shape[-1]]
    new_mri_ses = torch.zeros_like(dxa_ses)
    new_mri_ses[:,:,:mri_ses.shape[-2],:mri_ses.shape[-1]] = mri_ses
    catted = torch.cat([dxa_ses, new_mri_ses],dim=1)
    return catted


@ex.capture
def run_epoch(dl, model,dxa_encoder, mri_encoder, optim,USE_CUDA, train = True):
    dxa_encoder.eval()
    mri_encoder.eval()
    if train: model.train()
    else: model.eval()

    pbar = tqdm(dl)
    losses = []
    ang_diffs=[]
    t_x_diffs=[]
    t_y_diffs=[]
    ang_baselines=[]
    t_x_baselines=[]
    t_y_baselines=[]
    train_stats = {}
    for idx, sample in enumerate(pbar):
        optim.zero_grad()
        if USE_CUDA:
            for key in sample:
                sample[key] = sample[key].cuda()
        with torch.no_grad():
            dxa1_ses = F.normalize(dxa_encoder(sample['dxa_img1']),dim=1)
            dxa2_ses = F.normalize(dxa_encoder(sample['dxa_img2']),dim=1)
            mri1_ses = F.normalize(mri_encoder(sample['mri_img1']),dim=1)

        downsampled_dxa1 = F.interpolate(sample['dxa_img1'],size=(dxa1_ses.shape[-2],dxa1_ses.shape[-1]),mode='bicubic')
        downsampled_dxa2 = F.interpolate(sample['dxa_img2'],size=(dxa2_ses.shape[-2],dxa2_ses.shape[-1]),mode='bicubic')
        downsampled_mri1 = F.interpolate(sample['mri_img1'],size=(mri1_ses.shape[-2],mri1_ses.shape[-1]),mode='bicubic')
        downsampled_orig_imgs1  = cat_ses(downsampled_mri1,downsampled_dxa1)
        downsampled_orig_imgs2  = cat_ses(downsampled_mri1,downsampled_dxa2)
        #in1 = cat_ses(mri1_ses, dxa1_ses)
        #in2 = cat_ses(mri1_ses, dxa2_ses)
        angle1, t_x1, t_y1, coarse1 = model(dxa1_ses, mri1_ses)
        angle2, t_x2, t_y2, coarse2 = model(dxa2_ses, mri1_ses)
        t_x1 = t_x1*8; t_x2 = t_x2*8
        t_y1 = t_y1*8; t_y2 = t_y2*8
        coarse1[:,:2] = coarse1[:,:2]*8
        coarse2[:,:2] = coarse2[:,:2]*8


        #  angle loss
        ang_orig_loss = ((sample['angle_orig'] + sample['angle_aug'] + angle2)**2).mean()

        # translation loss
        orig_t = torch.stack([sample['t_x_orig'],sample['t_y_orig']], dim=1)
        delta_t = torch.stack([sample['t_x_aug'],sample['t_y_aug']],dim=1)
        orig_angle = sample['angle_orig']
        orig_rads = (orig_angle*np.pi)/180
        coses = torch.cos(orig_rads)
        sines = torch.sin(orig_rads)
        rot_mat = torch.stack([torch.stack([coses,-sines]),torch.stack([sines,coses])]).permute(2,0,1).float()
        target_t = orig_t + torch.bmm(rot_mat.float(), delta_t[:,:,None].float()).squeeze(-1)
        # import pdb; pdb.set_trace()
        t_x_loss = ((target_t[:,0]+t_x2)**2).mean()
        t_y_loss = ((target_t[:,1]+t_y2)**2).mean()
        # unwarp dxa 2
        # batch_idx=8
        # target_warped_dxa = rt_and_t(sample['dxa_img2'][batch_idx], 
        #                              sample['angle_orig'][batch_idx].item()+sample['angle_aug'][batch_idx].item(), 
        #                              target_t[batch_idx,0].item(), 
        #                              target_t[batch_idx,1].item())
        # mri_img = sample['mri_img1']
        # plt.imshow(red(target_warped_dxa[0,:mri_img.shape[-2],:mri_img.shape[-1]])+grayscale(mri_img[batch_idx,0]))
        # im_show()


        #t_x_loss = 
        # loss = ang_orig_loss + t_x_orig_loss + t_y_orig_loss
        loss = ang_orig_loss + t_x_loss + t_y_loss
        if train:
            loss.backward()
            optim.step()

        diff_ang = torch.abs(sample['angle_orig']+sample['angle_aug']+angle2).mean().item()
        diff_t_x = torch.abs(target_t[:,0]+t_x2).mean().item()
        diff_t_y = torch.abs(target_t[:,1]+t_y2).mean().item()
        ang_baselines.append(torch.abs(sample['angle_orig']+sample['angle_aug']).mean().item())
        t_x_baselines.append(torch.abs(target_t[:,0]+coarse2[:,0]).mean().item())
        t_y_baselines.append(torch.abs(target_t[:,1]+coarse2[:,1]).mean().item())

        losses.append(loss.item())
        ang_diffs.append(diff_ang)
        t_x_diffs.append(diff_t_x)
        t_y_diffs.append(diff_t_y)
        if idx > 10:
            agg_losses = np.mean(losses[-10:])
            agg_ang_diffs = np.mean(ang_diffs[-10:])
            agg_t_x_diffs = np.mean(t_x_diffs[-10:])
            agg_t_y_diffs = np.mean(t_y_diffs[-10:])
            agg_ang_baseline = np.mean(ang_baselines[-10:]) 
            agg_t_x_baseline = np.mean(t_x_baselines[-10:]) 
            agg_t_y_baseline = np.mean(t_y_baselines[-10:]) 

            pbar.set_description(f"loss:{agg_losses:.2f} ang: {agg_ang_diffs:.2f}({agg_ang_baseline:.2f}) t_x: {agg_t_x_diffs:.2f}({agg_t_x_baseline:.2f}) t_y:{agg_t_y_diffs:.2f}({agg_t_y_baseline:.2f})")

    stats = {'loss': np.mean(losses), 'ang_diffs':np.mean(ang_diffs),'t_x_diffs':np.mean(t_x_diffs),
                   't_y_diffs':np.mean(t_y_diffs), 'ang_baselines': np.mean(ang_baselines), 't_x_baselines':np.mean(t_x_baselines), 't_y_baselines':np.mean(t_y_baselines)}

    return stats

@ex.capture
def save_model(model, val_stats, epochs, LOAD_FROM_PATH):
    print(f'==> Saving Model Weights to {LOAD_FROM_PATH}')
    state = {'model_weights': model.state_dict(),
             'best_loss'        : val_stats['loss'],
             'epochs'           : epochs
            }
    if not os.path.isdir(LOAD_FROM_PATH):
        os.mkdir(LOAD_FROM_PATH)
    previous_checkpoints = glob.glob(LOAD_FROM_PATH + '/ckpt*.pt', recursive=True)
    for previous_checkpoint in previous_checkpoints:
       os.remove(previous_checkpoint)
    torch.save(state, LOAD_FROM_PATH + '/ckpt' + str(epochs) + '.pt')
    return

def rt_and_t(img, angle, t_x, t_y):
    return TF.affine(TF.rotate(img[None],angle,center=(0,0)),0,(t_y,t_x),1,(0,0))[0]

def t_and_rt(img, angle, t_x, t_y):
    return TF.rotate(TF.affine(img[None],0,(t_y,t_x),1,(0,0)),angle,center=(0,0))[0]

@ex.command(unobserved=True)
def try_alignment():
    train_dl, val_dl, test_dl = get_dataloaders()
    model,dxa_encoder, mri_encoder, best_loss, epochs = load_models()
    model.eval()
    for idx, sample in enumerate(val_dl):
        with torch.no_grad():
            dxa1_ses = F.normalize(dxa_encoder(sample['dxa_img1']),dim=1)
            dxa2_ses = F.normalize(dxa_encoder(sample['dxa_img2']),dim=1)
            mri1_ses = F.normalize(mri_encoder(sample['mri_img1']),dim=1)
            # angle1, t_x1, t_y1, coarse_t_x1, coarse_t_y1 = model(dxa1_ses, mri1_ses)
            # angle2, t_x2, t_y2, coarse_t_x1, coarse_t_y1 = model(dxa2_ses, mri1_ses)
        angle1, t_x1, t_y1, coarse1 = model(dxa1_ses, mri1_ses)
        angle2, t_x2, t_y2, coarse2 = model(dxa2_ses, mri1_ses)
        t_x1 = t_x1*8; t_x2 = t_x2*8
        t_y1 = t_y1*8; t_y2 = t_y2*8
        coarse1 = coarse1*8
        coarse2 = coarse2*8
        orig_t = torch.stack([sample['t_x_orig'],sample['t_y_orig']], dim=1)
        delta_t = torch.stack([sample['t_x_aug'],sample['t_y_aug']],dim=1)
        orig_angle = sample['angle_orig']
        orig_rads = (orig_angle*np.pi)/180
        coses = torch.cos(orig_rads)
        sines = torch.sin(orig_rads)
        rot_mat = torch.stack([torch.stack([coses,-sines]),torch.stack([sines,coses])]).permute(2,0,1).float()
        target_t = orig_t + torch.bmm(rot_mat.float(), delta_t[:,:,None].float()).squeeze(-1)
        print(coarse2, target_t)
        for batch_idx in range(sample['dxa_img1'].shape[0]):
            plt.figure(figsize=(20,10))
            plt.subplot(141)
            mri_img = sample['mri_img1']
            plt.imshow(red(sample['dxa_img1'][batch_idx,0,
                                              :mri_img.shape[-2],
                                              :mri_img.shape[-1]])+grayscale(sample['mri_img1'][batch_idx,0]))
            plt.title('Orig')
            plt.subplot(142)
            plt.title('Coarse')
            mri_img = sample['mri_img1']
            #import pdb; pdb.set_trace()
            target_warped_dxa = rt_and_t(sample['dxa_img1'][batch_idx],
                                         0,
                                         -coarse1[batch_idx,0],
                                         -coarse1[batch_idx,1])
            plt.imshow(red(target_warped_dxa[0,:mri_img.shape[-2],
                                             :mri_img.shape[-1]])+grayscale(mri_img[batch_idx,0]))
            plt.subplot(143)
            plt.title('Predict')
            mri_img = sample['mri_img1']
            target_warped_dxa = rt_and_t(sample['dxa_img1'][batch_idx],
                                         -angle1[batch_idx].item(),
                                         -t_x1[batch_idx].item(),
                                         -t_y1[batch_idx].item())
            plt.imshow(red(target_warped_dxa[0,:mri_img.shape[-2],
                                             :mri_img.shape[-1]])+grayscale(mri_img[batch_idx,0]))

            plt.subplot(144)
            plt.title('Predict')
            mri_img = sample['mri_img1']
            target_warped_dxa = rt_and_t(sample['dxa_img1'][batch_idx],
                                         sample['angle_orig'][batch_idx].item(),
                                         orig_t[batch_idx,0].item(),
                                         orig_t[batch_idx,1].item())
            plt.imshow(red(target_warped_dxa[0,:mri_img.shape[-2],
                                             :mri_img.shape[-1]])+grayscale(mri_img[batch_idx,0]))

            im_show()

            # ----------------------------
            # plt.imshow(rt_and_t(sample['dxa_img1'][batch_idx],
            # 0, sample['t_x_orig'][batch_idx], sample['t_y_orig'][batch_idx]))

@ex.automain
def main(_run):
    train_dl, val_dl, test_dl = get_dataloaders()
    model,dxa_encoder, mri_encoder, best_loss, epochs = load_models()
    optim = get_optim(model)
    while True:
        train_stats = run_epoch(train_dl,model, dxa_encoder, mri_encoder, optim, train=True)
        for key in train_stats:
            _run.log_scalar(f'training.{key}', train_stats[key])
        val_stats = run_epoch(val_dl,model, dxa_encoder, mri_encoder, optim, train=False)
        for key in val_stats:
            _run.log_scalar(f'validation.{key}', val_stats[key])

        if val_stats['loss'] < best_loss:
            best_loss = val_stats['loss']
            save_model(model,val_stats,epochs)
        epochs += 1

