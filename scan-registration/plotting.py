import sys, os, glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import cv2

sys.path.append('/users/rhydian/self-supervised-project')
from gen_utils import *

def show_matches(img1,img2, inliers1, inliers2):
    # pad both images to the same size

    if img1.shape[-2] < img2.shape[-2]:
        diff = img2.shape[-2]-img1.shape[-2]
        pad = diff/2
        show_img1 = np.concatenate([np.zeros((int(np.ceil(pad)),img1.shape[-1])), img1[0], np.zeros((int(np.floor(pad)),img1.shape[-1]))])
        show_img2 = img2[0]
        img1_hpad = np.ceil(pad)
        img2_hpad = 0
    elif img1.shape[-2] > img2.shape[-2]:
        diff = img1.shape[-2]-img2.shape[-2]
        pad = diff/2
        show_img2 = np.concatenate([np.zeros((int(np.ceil(pad)),img2.shape[-1])), 
                                               img2[0], np.zeros((int(np.floor(pad)),img2.shape[-1]))])
        show_img1 = img1[0]
        img1_hpad = 0
        img2_hpad = np.ceil(pad)
    else:
        diff = 0
        show_img2 = img2[0]
        show_img1 = img1[0]
    
    fig, ax = plt.subplots(figsize=(10,10))

    ax.imshow(grayscale(np.concatenate([show_img1,show_img2],axis=1)))
    colors=['r','g','y','b','cyan','orange','purple','white','magenta']
    for idx, point in enumerate(inliers1):
        ax.scatter(point[1],point[0]+img1_hpad,color=colors[idx%colors.__len__()],marker='x')
        match_point = inliers2[idx]
        ax.scatter(match_point[1]+img1.shape[-1],match_point[0]+img2_hpad,marker='x',color=colors[idx%colors.__len__()])
        ax.plot([match_point[1]+img1.shape[-1],point[1]],
                [match_point[0]+img2_hpad,point[0]+img1_hpad],color=colors[idx%colors.__len__()])
    return fig, ax

def show_transformed_keypoints(src_img, tgt_img, src_pts, est_tgt_pts, tgt_pts):
    ''' Show transformed fiducial markers '''
    if src_img.shape[-2] < tgt_img.shape[-2]:
        diff = tgt_img.shape[-2]-src_img.shape[-2]
        pad = diff/2
        show_src_img = np.concatenate([np.zeros((int(np.ceil(pad)),src_img.shape[-1])), src_img[0], np.zeros((int(np.floor(pad)),src_img.shape[-1]))])
        show_tgt_img = tgt_img[0]
        src_img_hpad = np.ceil(pad)
        tgt_img_hpad = 0
    elif src_img.shape[-2] > tgt_img.shape[-2]:
        diff = src_img.shape[-2]-tgt_img.shape[-2]
        pad = diff/2
        show_tgt_img = np.concatenate([np.zeros((int(np.ceil(pad)),tgt_img.shape[-1])), 
                                               tgt_img[0], np.zeros((int(np.floor(pad)),tgt_img.shape[-1]))])
        show_src_img = src_img[0]
        src_img_hpad = 0
        tgt_img_hpad = np.ceil(pad)
    else:
        diff = 0
        show_tgt_img = tgt_img[0]
        show_src_img = src_img[0]
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(grayscale(np.concatenate([show_src_img,show_tgt_img],axis=1)))
    colors=['r','g','y','b','cyan','orange','purple','white','magenta']
    labels=['left_arm', 'right_arm','spine_base','left_leg','right_leg']
    for idx, point in enumerate(src_pts):
        ax.scatter(point[1],point[0]+src_img_hpad,color=colors[idx%colors.__len__()],marker='x',label=labels[idx])
        match_point = est_tgt_pts[idx]
        true_point = tgt_pts[idx]
        ax.scatter(match_point[1]+src_img.shape[-1],match_point[0]+tgt_img_hpad,marker='x',color=colors[idx%colors.__len__()])
        ax.scatter(true_point[1]+src_img.shape[-1],true_point[0]+tgt_img_hpad,color=colors[idx%colors.__len__()],label=labels[idx])
        ax.plot([match_point[1]+src_img.shape[-1],point[1]],
                [match_point[0]+tgt_img_hpad,point[0]+src_img_hpad],color=colors[idx%colors.__len__()])
        ax.legend()
            
    return fig, ax

def show_warped_perspective(src_img, tgt_img, M):
        b, w, h = src_img.shape
        src_pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]) .reshape(-1,2) 
        tgt_pts = []
        for src_pt in src_pts:
            tgt_pt = (M @ np.concatenate([src_pt,[1]]))[:2]
            tgt_pts.append(tgt_pt)

        tgt_pts = np.array(tgt_pts)
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
        ax1.imshow(src_img[0],cmap='gray')
        ax1.axis('off')
        ax2.imshow(tgt_img[0],cmap='gray')
        ax2.axis('off')
        ax2.add_patch(Polygon(tgt_pts[:,[1,0]],ec='y',fc='none'))
        return fig,(ax1,ax2)


def show_transformed_segmentations(src_img, tgt_img, src_segment, tgt_segment, M):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
    ax1_img = grayscale(src_img[0])
    ax2_img = grayscale(tgt_img[0])
    b, w, h = src_img.shape
    src_pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]) .reshape(-1,2) 
    tgt_pts = []
    for src_pt in src_pts:
        tgt_pt = (M @ np.concatenate([src_pt,[1]]))[:2]
        tgt_pts.append(tgt_pt)
    tgt_pts = np.array(tgt_pts)

    color_fns = [red,green,blue,pink,yellow,brown]
    for idx in range(src_segment.shape[0]):
        ax1_img += color_fns[idx](src_segment[idx])
        ax2_img += color_fns[idx](tgt_segment[idx])

    ax1.imshow(ax1_img)
    ax2.imshow(ax2_img)
    ax2.add_patch(Polygon(tgt_pts[:,[1,0]],ec='y',fc='none'))
    return fig,(ax1,ax2)
