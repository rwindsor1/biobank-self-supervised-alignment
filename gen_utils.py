import random
import string
import os


import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.stats
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import torch.nn.functional as F

def red(x):
    '''
    Return RGB red image from 1d grayscale image 
    '''
    y = np.zeros_like(x)
    return np.stack([x,y,y], axis=-1)

def blue(x):
    '''
    Return RGB blue image from 1d grayscale image 
    '''
    y = np.zeros_like(x)
    return np.stack([y,y,x], axis=-1)

def green(x):
    '''
    Return RGB blue image from 1d grayscale image 
    '''
    y = np.zeros_like(x)
    return np.stack([y,x,y], axis=-1)

def pink(x):
    return 0.5*red(x)+0.5*blue(x)

def yellow(x):
    '''
    Return RGB yellow image from 1d grayscale image 
    '''
    return 0.5*red(x)+0.5*green(x)

def brown(x):
    return 0.5*blue(x)+0.5*green(x)



def grayscale(x):
    '''
    Return RGB image from 1d grayscale image with all channels gray
    '''
    return np.stack([x,x,x], axis=-1)

def balanced_l1w_loss(output, target):
    thresh = 0.01
    pos_val = (target > thresh).float()
    neg_val = (target <= thresh).float()
    pos_temp = (torch.sum(pos_val) + torch.sum(neg_val)) / (torch.sum(pos_val) + 1e-12)
    neg_temp = (torch.sum(pos_val) + torch.sum(neg_val)) / (torch.sum(neg_val) + 1e-12)
    pos_weight = pos_temp / (neg_temp + pos_temp)
    neg_weight = neg_temp / (neg_temp + pos_temp)
    se = torch.abs(output - target)
    # Weighted
    se[target >  thresh] *= pos_weight
    se[target <= thresh] *= neg_weight
    se[target == -1.0] = 0 # ignore_index = -1.0
    loss = torch.mean(se)
    return loss

def unbalanced_l1w_loss(output, target):
    se = torch.abs(output - target)
    # Weighted
    loss = torch.mean(se)
    return loss

def get_points(image, threshold):
    mask = image > threshold
    segmentation, no_labels = scipy.ndimage.label(mask)
    points = []
    for label_idx in range(no_labels):
        point = np.unravel_index(np.argmax(image*(segmentation==(label_idx+1))),image.shape)
        points.append(point)

    return points

def dice_and_iou(pred_logits,target):
    # pred logits are logits from network predictions, target is a binary map
    # FIX THIS!!!       
    tps = ((pred_logits > 0) * (target == 1)).sum()
    tns = ((pred_logits < 0) * (target == 0)).sum()
    fns = ((pred_logits < 0) * (target == 1)).sum()
    fps = ((pred_logits > 0) * (target == 0)).sum()


    return float(2*tps/(2*tps+fps+fns)), float(tps/(tps+fps+fns))

def find_best_alignment_scores(dxa_embeds: torch.Tensor,
                               mri_embeds: torch.Tensor) -> (torch.Tensor, np.array):
    '''
    Takes batches of spatial embeddings and finds the best alignments between 
    pairs using rotation and translation only

    Args
    ----
    dxa_embeds (torch.Tensor): a torch tensor of dimensions BxCxHxW representing
    the spatial embeddings of the dxa scans for the whole batch

    mri_embeds (torch.Tensor): a torch tensor of dimensions BxCxHxW representing
    the spatial embeddings of the mri scans for the whole batch

    Outputs
    -------
    scores (torch.Tensor): a list of dimensions BxB where the element at index
    (i,j) is the alignment score between dxa scan at index i in the batch and 
    the mri scan at index j

    alignments (np.array): an np array of size BxBx3. The vector at index 
    (i,j,:) contains elements (x,y,r) where (x,y) is the translation vector to
    align MRI scan indexed by j to DXA scan indexed by i and r is the rotation
    applied to the MRI.
    '''
    ROT_LOW = -45; ROT_HIGH = 45 # range of rotations to try
    ROT_GRANULARITY = 20 # number of rotations to try
    rotation_range = np.linspace(ROT_LOW*np.pi/180,ROT_HIGH*np.pi/180,20) 
    import time
    time1 = time.clock()
    # # pad out images to squares
    # F.pad(dxa_embeds, [])
    # rotation_matrices = torch.stack([torch.Tensor([[np.cos(theta),-np.sin(theta),0],
    #                                                [np.sin(theta),np.cos(theta),0]])
    #                                                for theta in rotation_range])
    # grid = F.affine_grid(rotation_matrices, mri_embeds.shape)
    # import pdb; pdb.set_trace()

    with torch.no_grad():
        for rotation_angle in rotation_range:
            rotated_mri_embeds = TF.rotate(mri_embeds, rotation_angle)
            for rotated_mri_embed in rotated_mri_embeds:
                align_scores = F.conv2d(dxa_embeds, rotated_mri_embed[None].cuda())
    time_elapsed = time.clock() - time1
    print(time_elapsed)



def spatial_embeds_dot_similarity(tensor1, tensor2):
    ''' 
    Finds the similarity between the two tensors representing spatial
    embeddings of both scan modalities. N.B. both should be normalised
    along the channels axis. The similarity is calculated by the
    dot product of each pair, averaged across the batch.
    $\sum{}

    Args
    ----
    dxa_embeds (torch.Tensor) : tensor of size BxCxHxW spatially encoding dxa scan
    mri_embeds (torch.Tensor) : tensor of size BxCxHxW spatially encoding mri scan

    Returns
    -------
    similarity (torch.Tensor) : a torch tensor of dimension B containing the similarity
    between the maps
    '''
    assert tensor1.shape == tensor2.shape, "Tensors should be the same shape"
    sim = (tensor1*tensor2).view(tensor1.shape[0],-1).sum(dim=1)
    return sim

def triplet_loss_dot_similarity(embeds_a, embeds_p, embeds_n, MARGIN):
    '''
    Triplet loss from spatial embeddings as defined by dot product between the
    two.

    Args
    ----
    embeds_a (torch.Tensor) : tensor of size BxCxHxW spatially encoding dxa scan
    embeds_p (torch.Tensor) : tensor of size BxCxHxW spatially encoding dxa scan
    embeds_n (torch.Tensor) : tensor of size BxCxHxW spatially encoding dxa scan

    '''
    positive_similarity = spatial_embeds_dot_similarity(embeds_a, embeds_p)
    negative_similarity = spatial_embeds_dot_similarity(embeds_a, embeds_p)
    import pdb; pdb.set_trace()
    loss = torch.clamp(negative_similarity + MARGIN - positive_similarity,min=0).mean()
    return loss


    


def make_segmentation_grid(dxa_vol, out_segmentation):
    grid = make_grid(torch.cat([.5*dxa_vol[:,0].unsqueeze(1)]*3, dim=1),nrow=5)
    for idx in range(out_segmentation.shape[1]):
        if idx == 0:
            part_grid = make_grid(torch.cat([out_segmentation[:,idx,None], 
                        torch.zeros_like(out_segmentation[:,idx,None]), 
                        torch.zeros_like(out_segmentation[:,idx,None])],dim=1),
                        nrow=5)     
        elif idx == 1:
            part_grid = make_grid(torch.cat([
                        torch.zeros_like(out_segmentation[:,idx,None]), 
                        out_segmentation[:,idx,None], 
                        torch.zeros_like(out_segmentation[:,idx,None])
                        ],dim=1),
                        nrow=5)     
        elif idx == 2:
            part_grid = make_grid(torch.cat([
                        torch.zeros_like(out_segmentation[:,idx,None]), 
                        torch.zeros_like(out_segmentation[:,idx,None]),
                        out_segmentation[:,idx,None]
                        ],dim=1),
                        nrow=5)     
        elif idx == 3:
            part_grid = make_grid(torch.cat([
                        torch.zeros_like(out_segmentation[:,idx,None]), 
                        0.5*out_segmentation[:,idx,None],
                        0.5*out_segmentation[:,idx,None]
                        ],dim=1),
                        nrow=5)     
        elif idx == 4:
            part_grid = make_grid(torch.cat([
                        0.5*out_segmentation[:,idx,None],
                        torch.zeros_like(out_segmentation[:,idx,None]), 
                        0.5*out_segmentation[:,idx,None]
                        ],dim=1),
                        nrow=5)     
        elif idx == 5:
            part_grid = make_grid(torch.cat([
                        0.5*out_segmentation[:,idx,None],
                        0.5*out_segmentation[:,idx,None],
                        torch.zeros_like(out_segmentation[:,idx,None])
                        ],dim=1),
                        nrow=5)     
                        
        grid += part_grid
    return grid

def im_show():
    # boilerplate to show current plt image as imgcat
    letters = string.ascii_letters
    temp_img_name = letters+'.png'
    ''.join(random.choice(letters) for i in range(10))
    plt.savefig(temp_img_name)
    os.system('imgcat ' + temp_img_name)
    os.remove(temp_img_name)
