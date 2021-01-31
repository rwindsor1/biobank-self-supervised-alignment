import os, sys, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvSequence, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, stride=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU())

    def forward(self,x):
        return self.layers(x)

class SpatialVGGMEncoder(nn.Module):
    def __init__(self, embedding_size=512, input_modes=1):
        super(SpatialVGGMEncoder, self).__init__()
        self.convs1 = ConvSequence(input_modes,   64,  kernel_size=(3,3), padding=1)
        self.convs2 = ConvSequence(64, 128,  kernel_size=(3,3), padding=1)
        self.convs3 = ConvSequence(128, 256, kernel_size=(3,3), padding=1) 
        self.convs4 = ConvSequence(256, embedding_size, kernel_size=(3,3), padding=1) 

    def forward(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1,kernel_size=2)
        x2 = self.convs2(x2)
        x3 = F.max_pool2d(x2,kernel_size=2)
        x3 = self.convs3(x3)
        x4 = F.max_pool2d(x3, kernel_size=2)
        x4 = self.convs4(x4)
        return x4

class SpatialVGGM2(nn.Module):
    def __init__(self, embedding_size=128, pooling='max', input_modes=1):
        super(SpatialVGGM2, self).__init__()
        assert pooling in ['max', 'average']
        if pooling == 'max':
            self.pool_function = F.max_pool2d
        elif pooling == 'average':
            self.pool_function = F.avg_pool2d


        self.convs1 = ConvSequence(input_modes,   64,  kernel_size=(3,3), padding=1)
        self.convs2 = ConvSequence(64, 128,  kernel_size=(3,3), padding=1)
        self.convs3 = ConvSequence(128, 256, kernel_size=(3,3), padding=1) 
        self.convs4 = ConvSequence(256, 512, kernel_size=(3,3), padding=1) 
        self.scan_lvl_convs = nn.Sequential(nn.Conv2d(512,512, kernel_size=(1,1)),
                                            nn.ReLU(),
                                            nn.Conv2d(512,embedding_size, kernel_size=(1,1)))

    def spatial_embeddings(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1,kernel_size=2)
        x2 = self.convs2(x2)
        x3 = F.max_pool2d(x2,kernel_size=2)
        x3 = self.convs3(x3)
        x4 = F.max_pool2d(x3, kernel_size=2)
        x4 = self.convs4(x4)
        x4 = self.scan_lvl_convs(x4)
        return x4


    # def spatial_embeddings_with_skips(self, x):
    #     x1 = self.convs1(x)
    #     x2 = F.max_pool2d(x1,kernel_size=2)
    #     x2 = self.convs2(x2)
    #     x3 = F.max_pool2d(x2,kernel_size=2)
    #     x3 = self.convs3(x3)
    #     x4 = F.max_pool2d(x3, kernel_size=2)
    #     x4 = self.convs4(x4)

    #     return x4,x3,x2,x1

    def forward(self,x):
        x = self.spatial_embeddings(x)
        x = self.pool_function(x,x.shape[2:])
        return x

    def transformed_spatial_embeddings(self, x):
        x = self.spatial_embeddings(x)
        return x

class ModeInvariantSpatialVGGM(nn.Module):
    def __init__(self, spatial_embedding_size=512, 
                 contrastive_embedding_size=128, pooling='max'):
        super().__init__()
        self.dxa_encoder = SpatialVGGMEncoder(embedding_size=spatial_embedding_size)
        self.mri_encoder = SpatialVGGMEncoder(embedding_size=spatial_embedding_size)

        self.contrastive_projector = nn.Sequential(nn.Conv2d(spatial_embedding_size,512, kernel_size=(1,1)),
                                                   nn.ReLU(),
                                                   nn.Conv2d(512,contrastive_embedding_size, kernel_size=(1,1)))
        if pooling == 'max':
            self.pool_function = F.max_pool2d
        elif pooling == 'average':
            self.pool_function = F.avg_pool2d

    def forward(self, dxa_scans, mri_scans):
        dxa_embeds = self.dxa_embed(dxa_scans)
        mri_embeds = self.mri_embed(mri_scans)
        return dxa_embeds, mri_embeds


    def dxa_spat_embed(self, x):
        x = self.dxa_encoder(x)
        x = F.normalize(x, dim=1)
        return x
        
    def mri_spat_embed(self, x):
        x = self.mri_encoder(x)
        x = F.normalize(x, dim=1)
        return x

    def mri_embed(self, x):
        x = self.mri_spat_embed(x)
        x = self.pool_function(x, x.shape[2:])
        x = self.contrastive_projector(x)
        return x

    def dxa_embed(self, x):
        x = self.dxa_spat_embed(x)
        x = self.pool_function(x, x.shape[2:])
        x = self.contrastive_projector(x)
        return x


if __name__ == '__main__':
    sys.path.append('/users/rhydian/self-supervised-project/datasets')
    from MidCoronalDataset import MidCoronalDataset
    ds = MidCoronalDataset(set_type='train', augment=True)
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    for idx in tqdm(range(len(ds))):
        dxa_img, mri_img = ds[idx]
        dxa = dxa_img[None].float()
        mri = mri_img[None].float()
        dxa_model = SpatialVGGM().eval()
        mri_model = SpatialVGGM().eval()
        mri_out = mri_model(mri)
        dxa_out = dxa_model(dxa)
        dxa_spat = dxa_model.spatial_embeddings(dxa)
        mri_spat = mri_model.spatial_embeddings(mri)
        import pdb; pdb.set_trace()
