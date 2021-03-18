import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinementRegressor(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(2*embedding_size, 512, 7, padding=3)
        self.conv2 = nn.Conv2d(512, 512, 7, padding=3)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 3)


    def get_coarse_align(self, dxa_ses, mri_ses):
        correlate_response = F.conv2d(dxa_ses, mri_ses)
        max_response, flat_idx = F.max_pool2d_with_indices(correlate_response,(correlate_response.shape[-2:]))
        ses_index = torch.stack([flat_idx.squeeze(-1).squeeze(-1).diag()//correlate_response.shape[-1],flat_idx.squeeze(-1).squeeze(-1).diag()%correlate_response.shape[-1]],dim=-1)

        return ses_index

        

    def forward(self, dxa_ses, mri_ses):
        ses_index = self.get_coarse_align(dxa_ses, mri_ses)
        if self.training:
            rand_augs = 5*(torch.rand(ses_index.size())-0.5)
            if ses_index.is_cuda:
                rand_augs = rand_augs.to(ses_index.get_device())

            ses_index = (ses_index.float() + rand_augs).long()
            ses_index[ses_index<0]=0
            ses_index[:,0][ses_index[:,0] > dxa_ses.size(-2) - mri_ses.size(-2)] = dxa_ses.size(-2) - mri_ses.size(-2)-1
            ses_index[:,1][ses_index[:,1] > dxa_ses.size(-1) - mri_ses.size(-1)] = dxa_ses.size(-1) - mri_ses.size(-1)-1

        coarse_aligned = self.realign_maps(dxa_ses,mri_ses, ses_index)
        conv_out = self.conv2(F.max_pool2d(torch.relu(self.conv1(coarse_aligned)),kernel_size=1))
        # global max pool
        pooled = F.max_pool2d(conv_out, kernel_size=conv_out.size()[-2:]).squeeze(-1).squeeze(-1)
        out = self.fc2(torch.relu(self.fc1(pooled)))
        angle = out[:,0]
        t_x = out[:,1] + ses_index[:,0]
        t_y = out[:,2] + ses_index[:,1]
        return angle, t_x, t_y, ses_index

    def realign_maps(self, dxa_ses, mri_ses, ses_index):
        new_mri_ses = torch.zeros_like(dxa_ses)
        for idx in range(new_mri_ses.size(0)):
            new_mri_ses[idx,:, 
                        ses_index[idx,0].item():ses_index[idx,0].item()+mri_ses.size(-2),
                        ses_index[idx,1].item():ses_index[idx,1].item()+mri_ses.size(-1)] = mri_ses[idx]

        coarse_aligned = torch.cat([dxa_ses, new_mri_ses], dim=1)
        return coarse_aligned
