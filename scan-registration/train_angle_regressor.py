import sys, os, glob
sys.path.append('/users/rhydian/self-supervised-project')
from gen_utils import *
import torch
from registration_dataset import RegistrationTrainingDataset
from registration_model import vgg16_bn
from torch.utils.data import DataLoader
from sacred import Experiment
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from sacred.observers import MongoObserver
import torchvision.transforms.functional as TF

ex=Experiment('TrainAngleRegressor')
ex.captured_out_filter = lambda captured_output: "output capturing turned off."
ex.observers.append(MongoObserver(url='login1.triton.cluster:27017', db_name='SelfSupervisedProject'))

@ex.config
def config():
    BATCH_SIZE=10
    NUM_WORKERS=10
    LOAD_FROM_PATH='/users/rhydian/self-supervised-project/model_weights/angle_regressor4'
    USE_CUDA=True
    ADAM_BETAS = (0.9,0.999)
    LR=0.0001
    SAVE_IMAGES=True
    ZERO_PARAMETER=1
    ROT_HIGH = 10
    ROT_LOW =10
    T_HIGH =10
    T_LOW =10

@ex.capture
def get_dataloaders(BATCH_SIZE, NUM_WORKERS,ROT_HIGH,ROT_LOW,T_HIGH,T_LOW):
    train_ds = RegistrationTrainingDataset(set_type='train', augment_first=True,  rot_high=ROT_HIGH,rot_low=ROT_LOW,trans_high=T_HIGH,trans_low=T_LOW )
    val_ds = RegistrationTrainingDataset(  set_type='val',   augment_first=False, rot_high=ROT_HIGH,rot_low=ROT_LOW,trans_high=T_HIGH,trans_low=T_LOW)
    test_ds = RegistrationTrainingDataset( set_type='test',  augment_first=False, rot_high=ROT_HIGH,rot_low=ROT_LOW,trans_high=T_HIGH,trans_low=T_LOW)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    return train_dl, val_dl, test_dl


@ex.capture
def load_model(LOAD_FROM_PATH, USE_CUDA):
    model = vgg16_bn(num_classes=3)
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
    return model, best_loss, epochs

@ex.capture
def train_epoch(dl, model,optim,use_cuda, zero_parameter):
    model.train()
    pbar = tqdm(dl)
    true_angles_off = []
    delta_angles_off = []
    true_xys_off = []
    delta_xys_off = []
    losses = []
    zero_losses=[]
    for idx, sample in enumerate(pbar):
        optim.zero_grad()
        if use_cuda:
            for key in sample:
                sample[key] = sample[key].cuda()
        out1 = model(sample['img1'])
        out2 = model(sample['img2'])
        delta_angle = sample['angle2']
        delta_t = sample['translations2']
        pred_angles = out2[:,0] - out1[:,0]
        delta_rads = (delta_angle*np.pi)/180
        coses = torch.cos(delta_rads)
        sines = torch.sin(delta_rads)
        rot_mat = torch.stack([torch.stack([coses,-sines]),torch.stack([sines,coses])]).permute(2,0,1).float()
        #rot_mat = torch.stack([torch.stack([1,-delta_rands]),torch.stack([delta_rads,1])]).permute(2,0,1).float()
        pred_t = out2[:,1:] - torch.bmm(rot_mat,out1[:,1:,none]).squeeze(-1)
        angle_loss = (delta_angle - pred_angles).abs().mean()
        t_loss = (delta_t - pred_t).norm(dim=1).mean()
        zero_loss = out1.abs().mean()
        loss = angle_loss + t_loss + zero_parameter*zero_loss
        loss.backward()
        optim.step()
        #metrics
        delta_angle_err = angle_loss.item()
        delta_xy_err = t_loss.item()
        true_angle_err = (out1[:,1] - sample['angle1']).abs().mean()
        true_xy_err = (out1[:,2:] - sample['translations1']).norm(dim=1).mean()
        pbar.set_description(f"loss: {loss.item():.5}, angle(true): {true_angle_err:.5} error_angle(d): {delta_angle_err:.5} xy(true): {true_xy_err:.5} xy(delta):{delta_xy_err:.5} avg angle:{sample['angle2'].abs().mean():.4}, avg t: {sample['translations2'].norm(dim=1).mean():.4}")
        losses.append(loss.item())
        delta_xys_off.append(delta_xy_err)
        delta_angles_off.append(delta_angle_err)
        true_xys_off.append(true_xy_err.item())
        true_angles_off.append(true_angle_err.item())
        zero_losses.append(zero_loss.item())
    return {'loss':np.mean(losses),'angle_off_true':np.mean(true_angles_off),'angle_off_delta':np.mean(delta_angles_off),
            'xy_off_true':np.mean(true_xys_off),'xy_off_delta':np.mean(delta_xys_off),'zero_losses':np.mean(zero_losses)}

@ex.capture
def save_models(model, val_stats, epochs, LOAD_FROM_PATH):
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

@ex.capture
def val_epoch(dl,model,USE_CUDA,_run):
    model.eval()
    pbar = tqdm(dl)
    true_angles_off = []
    delta_angles_off = []
    true_xys_off = []
    delta_xys_off = []
    losses = []
    all_angles = []
    all_translations = []
    for idx, sample in enumerate(pbar):
        with torch.no_grad():
            if USE_CUDA:
                for key in sample:
                    sample[key] = sample[key].cuda()
            out1 = model(sample['img1'])
            all_angles += out1[:,0].cpu().tolist()
            all_translations += out1[:,1:].cpu().tolist()
            out2 = model(sample['img2'])
            delta_angle = sample['angle2']
            delta_t = sample['translations2']
            pred_angles = out2[:,0] - out1[:,0]
            delta_rads = (delta_angle*np.pi)/180
            coses = torch.cos(delta_rads)
            sines = torch.sin(delta_rads)
            rot_mat = torch.stack([torch.stack([coses,-sines]),torch.stack([sines,coses])]).permute(2,0,1).float()
            pred_t = out2[:,1:] - torch.bmm(rot_mat,out1[:,1:,None]).squeeze(-1)
            angle_loss = (delta_angle - pred_angles).abs().mean()
            t_loss = (delta_t - pred_t).norm(dim=1).mean()
            loss = angle_loss + t_loss
            #metrics
            delta_angle_err = angle_loss.item()
            delta_xy_err = t_loss.item()
            true_angle_err = (out1[:,1] - sample['angle1']).abs().mean()
            true_xy_err = (out1[:,2:] - sample['translations1']).norm(dim=1).mean()
            pbar.set_description(f"loss: {loss.item():.5}, angle(true): {true_angle_err:.5} error_angle(d): {delta_angle_err:.5} xy(true): {true_xy_err:.5} xy(delta):{delta_xy_err:.5} avg angle:{sample['angle2'].abs().mean():.4}, avg t: {sample['translations2'].norm(dim=1).mean():.4}")
            losses.append(loss.item())
            delta_xys_off.append(delta_xy_err)
            delta_angles_off.append(delta_angle_err)
            true_xys_off.append(true_xy_err.item())
            true_angles_off.append(true_angle_err.item())

    all_angles = np.array(all_angles)
    all_translations = np.array(all_translations)
    print(f'Mean angle {np.median(all_angles)} +- {np.std(all_angles)}')
    print(f'transx: {np.median(all_translations[:,0])} + {np.std(all_translations[:,0])}')
    print(f'transy: {np.median(all_translations[:,1])} + {np.std(all_translations[:,1])}')
    return {'loss':np.mean(losses),'angle_off_true':np.mean(true_angles_off),'angle_off_delta':np.mean(delta_angles_off),
            'xy_off_true':np.mean(true_xys_off),'xy_off_delta':np.mean(delta_xys_off)}
    # make visuals




@ex.capture
def make_visual_examples(dl,model, SAVE_IMAGES):
    model.eval()
    for idx, sample in enumerate(dl):
        with torch.no_grad():
            out1 = model(sample['img1'])
            out2 = model(sample['img2'])
        delta_angle = -(out2[:,0] - out1[:,0])
        delta_t = -(out2[:,1:] - out1[:,1:])
        gt_delta_angle = -sample['angle2']
        gt_delta_t = -sample['translations2']
        for batch_idx in tqdm(range(sample['img1'].shape[0])):
            dxas = sample['img2'][batch_idx,:2][None]
            dxas = TF.affine(dxas,delta_angle[batch_idx].item(),delta_t[batch_idx].tolist(),1,[0,0])
            t_img1= torch.cat([dxas[0], sample['img2'][batch_idx,2:]])
            gt_dxas = sample['img2'][batch_idx,:2][None]
            gt_dxas = TF.affine(gt_dxas,gt_delta_angle[batch_idx].item(),gt_delta_t[batch_idx].tolist(),1,[0,0])
            gt_t_img1= torch.cat([gt_dxas[0], sample['img1'][batch_idx,2:]])
            plt.subplot(151)
            plt.title('src')
            plt.imshow(grayscale(sample['img1'][batch_idx,2])+red(sample['img1'][batch_idx,0]))
            plt.subplot(152)
            plt.title('augmented')
            plt.imshow(grayscale(sample['img2'][batch_idx,2])+red(sample['img2'][batch_idx,0]))
            plt.subplot(153)
            plt.title('pred')
            plt.imshow(grayscale(t_img1[2])+red(t_img1[0]))
            plt.subplot(154)
            plt.title('diff')
            plt.imshow(red(sample['img1'][batch_idx,0])+blue(t_img1[0]))
            plt.subplot(155)
            plt.title('true diff')
            plt.imshow(red(sample['img1'][batch_idx,0])+blue(gt_t_img1[0]))
            #plt.subplot(144)
            #plt.title('gt')
            #plt.imshow(grayscale(gt_t_img1[2])+red(gt_t_img1[0]))
            k = idx*10 + batch_idx
            plt.suptitle(f'theta: {out1[batch_idx,0].item():.4} t: {out1[batch_idx,1:].tolist()}, delta: {delta_angle[batch_idx].item():.3}, true:{gt_delta_angle[batch_idx].item():.3},\n trans:{delta_t[batch_idx]}, true:{gt_delta_t[batch_idx]}',fontsize=8)
            if SAVE_IMAGES:
                plt.savefig(f'examples/example_{k}')
            else:
                plt.savefig(f'temp.png')
                os.system('imgcat temp.png')
            plt.close('all')
        if k > 95: break

@ex.command(unobserved=True)
def try_alignment():
    target_angle=0
    target_translation=torch.Tensor([0,0])
    train_dl, val_dl, test_dl = get_dataloaders()
    model, best_loss, epochs = load_model()
    model.eval()
    for idx, sample in enumerate(val_dl):
        with torch.no_grad():
            angles = np.random.random(sample['img1'].shape[0])*6-3
            translations = np.random.random((sample['img1'].shape[0],2))*6-3
            img1 = sample['img1']
            for batch_idx in range(sample['img1'].shape[0]):
                img1[batch_idx,:2] = TF.affine(img1[batch_idx,:2][None],angles[batch_idx], translations[batch_idx].tolist(),1,[0,0])[None]
            out1 = model(img1)
            for batch_idx in range(sample['img1'].shape[0]):
                orig_img = sample['img1'][batch_idx]
                dxas=sample['img1'][batch_idx,:2][None]
                pred_angle=out1[batch_idx,0]
                translation=out1[batch_idx,1:]
                move_angle=-pred_angle+target_angle
                move_translation=-translation+target_translation
                dxas = TF.affine(dxas,move_angle.item(),move_translation.tolist(),1,[0,0])
                t_img1= torch.cat([dxas[0], sample['img1'][batch_idx,2:]])
                plt.subplot(121)
                plt.title('original')
                plt.imshow(grayscale(orig_img[2])+red(orig_img[0]))
                plt.subplot(122)
                plt.title('transformed')
                plt.imshow(grayscale(t_img1[2])+red(t_img1[0]))
                plt.suptitle(f"angle {move_angle:.3} {move_translation}")
                plt.savefig(f'temp.png')
                os.system('imgcat temp.png')

                orig_img = sample['img1']

@ex.capture
def get_optim(model, LR, ADAM_BETAS):
    optim = Adam(model.parameters(), lr=LR, betas=ADAM_BETAS)
    return optim

@ex.command(unobserved=True)
def test():
    train_dl, val_dl, test_dl = get_dataloaders()
    model, best_loss, epochs = load_model()
    optim = get_optim(model)
    make_visual_examples(test_dl, model)
    test_stats = val_epoch(test_dl,model)


@ex.automain
def main(_run):
    train_dl, val_dl, test_dl = get_dataloaders()
    model, best_loss, epochs = load_model()
    optim = get_optim(model)
    while True:
        for i in range(10):
            train_stats = train_epoch(train_dl,model,optim)
        for key in train_stats:
            _run.log_scalar(f'training.{key}', train_stats[key])
        val_stats = val_epoch(val_dl,model)
        for key in val_stats:
            _run.log_scalar(f'validation.{key}', val_stats[key])

        if val_stats['loss'] < best_loss:
            best_loss = val_stats['loss']
            save_models(model,val_stats,epochs)
        epochs += 1

