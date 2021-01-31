import sys, os, glob
sys.path.append('/users/rhydian/self-supervised-project')
from models.SpatialVGGM import SpatialVGGM
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(UpConvSequence, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False),
                                    nn.ReLU())
                                    

    def forward(self,x):
        return self.layers(x)

class SegmentationUnetLowRes(nn.Module):
    def __init__(self, SpatialEncoder : SpatialVGGM, use_skips=True, nclasses=1):
        super(SegmentationUnetLowRes, self).__init__()
        self.SpatialEncoder = SpatialEncoder

        self.up_convs1 = UpConvSequence(128, 128, (3,3), 1)
        self.up_convs2 = UpConvSequence(128, 64, (3,3), 1)
        self.up_convs3 = UpConvSequence(64, 32, (3,3), 1)
        self.last_conv = nn.Conv2d(32, nclasses,(3,3),padding=1)
            

    def get_transformed_spatial_embeddings(self, x):
        x = self.SpatialEncoder.module.transformed_spatial_embeddings(x)
        x = F.normalize(x, dim=1)
        return x

    def forward(self, x):
        sizes = ((torch.tensor(x.shape)/4).int()[2:], (torch.tensor(x.shape)/2).int()[2:], torch.tensor(x.shape[2:]))
        # import pdb; pdb.set_trace()
        x = self.get_transformed_spatial_embeddings(x)

        x = self.up_convs1(x)
        x = F.interpolate(x, tuple(sizes[0].tolist()))
        x = self.up_convs2(x)
        x = F.interpolate(x, tuple(sizes[1].tolist()))
        x = self.up_convs3(x)
        x = F.interpolate(x, tuple(sizes[2].tolist()))
        x = self.last_conv(x)
        return x




if __name__ == '__main__':
    from train_contrastive_negative_mining import load_models
    PRETRAINED_MODELS_PATH = '../model_weights/ContrastiveModelsFixedAug_MaxPool'
    from datasets.SegmentationDatasets import MRISlicesSegmentationDataset
    ds = MRISlicesSegmentationDataset(all_root='/scratch/shared/beegfs/rhydian/UKBiobank/',augment=False,blanking_augment=False)
    dxa_model, mri_model, _, _ = load_models(PRETRAINED_MODELS_PATH, 'max', use_cuda=False)
    # train_dl, val_dl, test_dl = get_dataloaders(1,0)
    unet = SegmentationUnetLowRes(dxa_model)
    sample = ds[0]
    out = unet(sample['dxa_vol'][None])
    import pdb; pdb.set_trace()
    

