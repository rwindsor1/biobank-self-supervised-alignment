'''
Function definitions to perform automatic scan correlation using our method
Rhydian Windsor 18/02/20
'''

import sys, os, glob
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import math
import torchvision.transforms.functional as TF
sys.path.append('/users/rhydian/self-supervised-project')
from gen_utils import *
from torchgeometry.core import get_rotation_matrix2d

def get_4d_corr(se1, se2):
    # get 4d correlation between spatial feature maps
    b1, c1, h1, w1 = se1.size()
    b2, c2, h2, w2 = se2.size()
    corr = torch.bmm(se1.permute(0,2,3,1).view(b1,h1*w1,c1),se2.view(b2, c2, h2*w2))
    corr = corr.view(b1, h1, w1, h2, w2)
    return corr

def fit_RANSAC(src_pts, tgt_pts, tol=5):
    src_pts = np.array(src_pts)[:,np.newaxis]
    tgt_pts = np.array(tgt_pts)[:,np.newaxis] 
    M, mask = cv2.findHomography(src_pts, tgt_pts, cv2.RANSAC,tol)
    src_inliers = np.array([src_point[0] for src_idx, src_point in enumerate(src_pts) if mask.ravel()[src_idx]])
    tgt_inliers = np.array([tgt_point[0] for tgt_idx, tgt_point in enumerate(tgt_pts) if mask.ravel()[tgt_idx]])
    return src_inliers, tgt_inliers,M

def resample_point(src_img, tgt_img, point):
    # resample points from one image frame to another
    hs, ws = src_img.size()
    ht, wt = tgt_img.size()
    resampled_point = [None, None]
    resampled_point[0] = np.round(point[0]*ht/hs).astype(np.int)
    resampled_point[1] = np.round(point[1]*wt/ws).astype(np.int)
    return resampled_point

def correlate_point(corr, pt=(30,30),tgt='a', softmax=True):
    #  get correlation map in other image correlating to a point `pt`
    assert tgt in ['a','b']
    if tgt == 'a':
        corr_point_map = corr[:,pt[0],pt[1],:,:]
    else:
        corr_point_map = corr[:,:,:,pt[0],pt[1]]
    temperature = 0.1
    if softmax:
        b, h, w = corr_point_map.size()
        corr_point_map = F.softmax(corr_point_map.view(b,h*w)/temperature,dim=1).view(b,h,w)
    
    return corr_point_map

def transform_points(src_pts, M):
        tgt_pts = []
        for src_pt in src_pts:
            tgt_pt = (M @ np.concatenate([src_pt,[1]]))[:2]
            tgt_pts.append(tgt_pt)
        return np.array(tgt_pts)

def register_scans(scan1,scan2, t, angle):
    scan1 = F.pad(scan1, (int(t[1]),0,int(t[0]),0))
    hdiff = scan2.shape[-2] - scan1.shape[-2]
    wdiff = scan2.shape[-1] - scan1.shape[-1]
    scan1 = F.pad(scan1, (0,wdiff,0,hdiff))
    scan1 = TF.rotate(scan1, angle,center=(0,0))
    return scan1


def find_best_rigid(src_pts, tgt_pts, allow_scaling=True):
    src_centroid = np.mean(src_pts,axis=0)
    tgt_centroid = np.mean(tgt_pts,axis=0)
    covar = (src_pts - src_centroid).T@(tgt_pts - tgt_centroid)
    U,S,V = np.linalg.svd(covar)
    R = V@U.T
    pred_exp_norm = np.linalg.norm((R@(src_pts-src_centroid).T).T,axis=1).mean()
    tgt_norm = np.linalg.norm(tgt_pts-tgt_centroid,axis=1).mean()
    if allow_scaling:
        scaling = tgt_norm/pred_exp_norm
    else:
        scaling = 1
    angle = math.atan2(R[1,0], R[0,0])
    t = tgt_centroid - scaling*R@src_centroid
    M = np.concatenate([scaling*R,t[:,None]],axis=1)
    M = np.concatenate([M,np.zeros(3)[None]])
    M[-1,-1] = 1
    angle = angle*180/np.pi
    
    return M, scaling, angle, t


def find_matching_error(src_pts, tgt_pts, M):
    est_tgt_pts = []
    for src_pt in src_pts:
        est_tgt_pt = (M @ np.concatenate([src_pt,[1]]))[:2].tolist()
        est_tgt_pts.append(est_tgt_pt)

    errors = np.linalg.norm(np.array(est_tgt_pt) - tgt_pts, axis=-1)
    avg_error = np.mean(errors)
    return avg_error


def get_most_discriminative_points(points, corr_map, src_img, tgt_img, num_points=80, use_cyclic=True):
    scored_points = []
    for idx, point in enumerate(points):
        ses_point = resample_point(src_img[0],corr_map[:,:,0,0], point)
        # get correlation map
        corr_point_map = correlate_point(corr_map[None], ses_point, tgt='a', softmax=True)
        # find argmax of correlation map
        tgt_ses_pt = [int(corr_point_map[0].argmax() // corr_point_map.shape[2]), int(corr_point_map[0].argmax() % corr_point_map.shape[2])]
        if use_cyclic:
            other_corr_point_map = correlate_point(corr_map[None], tgt_ses_pt, tgt='b', softmax=True)
            cyclic_score = other_corr_point_map[0,ses_point[0], ses_point[1]]
            score = corr_point_map.max()*cyclic_score
        else:
        # get correlation at this point
            score = corr_point_map.max()
        # resample argmax correlation point
        corres_point = resample_point(corr_map[0,0,:,:], tgt_img[0], tgt_ses_pt)
        # save to array
        scored_points.append([point, corres_point, corr_point_map.max(), ses_point])

    # sort and choose most discriminative point
    scored_points.sort(key=lambda x: -x[2])
    return [x[0] for x in scored_points[:num_points]],[x[1] for x in scored_points[:num_points]], [x[2] for x in scored_points[:num_points]]

def get_dense_correspondances(img1,img2,enc_model1,enc_model2,use_cuda, upsample_factor,rotation_range=(-2,2),tries=1):
    enc_model1.eval();enc_model2.eval()
    img1=img1[None];img2=img2[None]
    if use_cuda:
        img1 = img1.cuda()
        img2 = img2.cuda()
    else:
        img1=img1.cpu();img2=img2.cpu()
        enc_model1 = enc_model1.module.cpu()
        enc_model2 = enc_model2.module.cpu()

    with torch.no_grad():
        ses1 = enc_model1(img1)
        ses2 = enc_model2(img2)

    if upsample_factor != 1:
        ses1 = F.interpolate(ses1, scale_factor=upsample_factor, mode='bicubic',align_corners=False)
        ses2 = F.interpolate(ses2, scale_factor=upsample_factor, mode='bicubic',align_corners=False)
    angles=[]
    image_indexes=[]
    ses1 = F.normalize(ses1, dim=1)
    ses2 = F.normalize(ses2, dim=1)
    best_score = -9999
    best_angle = 0
    best_idx = None
    for angle in np.linspace(rotation_range[0],rotation_range[1],tries):
        if ses1.shape[-1] > ses2.shape[-1]:
            correlate_response = F.conv2d(ses1,TF.rotate(ses2,angle,center=(0,0)))
            max_response, flat_idx = F.max_pool2d_with_indices(correlate_response,(correlate_response.shape[-2:]))
            ses_index = [flat_idx.item()//correlate_response.shape[-1],flat_idx.item()%correlate_response.shape[-1]]
        else:
            correlate_response = F.conv2d(ses2,TF.rotate(ses1,angle,center=(0,0)))
            max_response, flat_idx = F.max_pool2d_with_indices(correlate_response,(correlate_response.shape[-2:]))
            ses_index = [flat_idx.item()//correlate_response.shape[-1],flat_idx.item()%correlate_response.shape[-1]]
        img_idx = resample_point(ses2[0,0],img2[0,0],ses_index)
        if max_response > best_score:
            best_score = max_response
            best_angle = angle
            best_idx = img_idx

    radians = -best_angle*3.1419/180
    M = [[np.cos(radians),np.sin(radians),best_idx[0]],
         [-np.sin(radians),np.cos(radians),best_idx[1]],
         [0,0,1]]
    M = np.array(M)
    return M, best_idx, best_angle

def get_salient_point_correspondances(img1,img2,model1,model2,threshold,use_ransac,ransac_tolerance,use_cuda):
    if use_cuda:
        img1 = img1.cuda()[None]
        img2 = img2.cuda()[None]
    else:
        img1=img1.cpu();img2=img2.cpu()
        model1 = model1.module.cpu()
        model2 = model2.module.cpu()
    with torch.no_grad():
        ses1 = F.normalize(model1(img1),dim=1)
        ses2 = F.normalize(model2(img2),dim=1)
    b1, c1, w1, h1 = ses1.size()
    b2, c2, w2, h2 = ses2.size()
    corr4d = torch.bmm(ses1.view(b1,c1,w1*h1).permute(0,2,1),ses2.view(b2,c2,w2*h2)).view(b1, w1, h1, w2, h2)
    corr4d_points = corr4d.view(w1*h1,w2*h2)
    scores, matches = torch.topk(corr4d_points,2)
    good = []
    for idx in range(len(matches)):
        if threshold*scores[idx,0] > scores[idx,1]:
            if resample_point(ses2[0,0], img2[0,0],[matches[idx,0].item()// h2, matches[idx,0].item() % h2]) not in [x[1] for x in good]:
                good.append([resample_point(ses1[0,0], img1[0,0],[idx // h1, idx % h1]),
                             resample_point(ses2[0,0], img2[0,0],[matches[idx,0].item()// h2, matches[idx,0].item() % h2])])
    good = np.array(good)
    if use_ransac:
        M, mask = cv2.findHomography(good[:,0],good[:,1],cv2.RANSAC, ransac_tolerance)
    ransac_good = good[[idx for idx, val in enumerate(mask) if val==1]]
    M, scaling, angle, t = find_best_rigid(ransac_good[:,0],ransac_good[:,1],allow_scaling=False)
    return ransac_good, np.linalg.inv(M), angle, t

def RefinementNetwork(dxa_img, mri_img, dxa_model, mri_model,threshold, ransac_tolerance, use_cuda, refinement_model):
    ransac_good, M, coarse_angle, coarse_t = get_salient_point_correspondances(dxa_img, mri_img,dxa_model,mri_model,threshold,True,ransac_tolerance,use_cuda)
    refinement_model.eval()
    # get matrix to warp DXA onto MRI
    new_M=np.linalg.inv(M); new_M[0,1],new_M[1,0]=new_M[1,0],new_M[0,1]; new_M[1,2],new_M[0,2]=new_M[0,2],new_M[1,2]
    warped_dxa = cv2.warpPerspective(np.array(255*dxa_img[0].permute(1,2,0)).astype('uint8'),
                                     new_M,(mri_img.shape[-1], mri_img.shape[-2]))
    warped_dxa = torch.Tensor(warped_dxa).permute(2,0,1).float()[None]/255

    refinement_input = torch.cat([warped_dxa,mri_img],dim=1)
    if use_cuda:
        refinement_input = refinement_input.cuda()
    with torch.no_grad():
        angle, t_x, t_y = refinement_model(refinement_input)[0].tolist()
    warped_dxa2 = TF.affine(warped_dxa, -angle, [-t_x,-t_y],1,[0,0])

    R=get_rotation_matrix2d(torch.Tensor([mri_img.shape[-2]/2, mri_img.shape[-1]/2])[None],torch.Tensor([angle]),torch.Tensor([1]))[0]
    R[0,2] += t_y
    R[1,2] += t_x
    # get homography matrix from these predictions
    # plt.subplot(121)
    # plt.imshow(red(warped_dxa[0,0])+grayscale(mri_img[0,0]))
    # for pt in src_pts:
    #     plt.scatter(pt[1],pt[0])
    # plt.subplot(122)
    # plt.imshow(red(warped_dxa[0,0])+grayscale(mri_img[0,0]))
    # for pt in src_pts:
    #     pt = R@np.array(pt + [1])
    #     plt.scatter(pt[1],pt[0])
    # plt.subplot(122)
    #plt.imshow(red(warped_dxa2[0,0])+grayscale(mri_img[0,0]))
    # plt.savefig('test.png')
    # os.system('imgcat test.png')
    # plt.close('all')
    #import pdb; pdb.set_trace()
    return M,R, angle, t_x, t_y, coarse_angle, coarse_t[0], coarse_t[1]




    #cv2.warpPerspective(np.array(255*sample['dxa_img'][0]).astype('uint8'),new_M,
    #                   (sample['mri_img'].shape[-1], sample['mri_img'].shape[-2]))
    
