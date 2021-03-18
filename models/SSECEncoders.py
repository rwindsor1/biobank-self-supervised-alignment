import sys, os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ConvSequence3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvSequence3d, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, stride=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU())

    def forward(self,x):
        return self.layers(x)

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

class VGGEncoder(nn.Module):
    def __init__(self, embedding_size=512, input_modes=1):
        super(VGGEncoder, self).__init__()
        self.embedding_size = embedding_size
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

    def forward_with_skips(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1,kernel_size=2)
        x2 = self.convs2(x2)
        x3 = F.max_pool2d(x2,kernel_size=2)
        x3 = self.convs3(x3)
        x4 = F.max_pool2d(x3, kernel_size=2)
        x4 = self.convs4(x4)
        return x4, x3, x2, x1

class VGGEncoder(nn.Module):
    def __init__(self, embedding_size=512, input_modes=1):
        super(VGGEncoder, self).__init__()
        self.embedding_size = embedding_size
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

    def forward_with_skips(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1,kernel_size=2)
        x2 = self.convs2(x2)
        x3 = F.max_pool2d(x2,kernel_size=2)
        x3 = self.convs3(x3)
        x4 = F.max_pool2d(x3, kernel_size=2)
        x4 = self.convs4(x4)
        return x4, x3, x2, x1

class VGGEncoderPoolAll(nn.Module):
    def __init__(self, embedding_size=512, input_modes=1):
        super(VGGEncoderPoolAll, self).__init__()
        self.embedding_size = embedding_size
        self.convs1 = ConvSequence(input_modes,   64,  kernel_size=(3,3), padding=1)
        self.convs2 = ConvSequence(64, 128,  kernel_size=(3,3), padding=1)
        self.convs3 = ConvSequence(128, 256, kernel_size=(3,3), padding=1) 
        self.convs4 = ConvSequence(256, 256, kernel_size=(3,3), padding=1)
        self.convs5 = ConvSequence(256, embedding_size, kernel_size=(1,1), padding=0)

    def forward(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1,kernel_size=2)
        x2 = self.convs2(x2)
        x3 = F.max_pool2d(x2,kernel_size=2)
        x3 = self.convs3(x3)
        x4 = F.max_pool2d(x3, kernel_size=2)
        x4 = self.convs4(x4)
        x5 = F.max_pool2d(x4, kernel_size=(x4.shape[-2],x4.shape[-1]))
        x5 = self.convs5(x5)
        return x5

    def forward_with_skips(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1,kernel_size=2)
        x2 = self.convs2(x2)
        x3 = F.max_pool2d(x2,kernel_size=2)
        x3 = self.convs3(x3)
        x4 = F.max_pool2d(x3, kernel_size=2)
        x4 = self.convs4(x4)
        return x4, x3, x2, x1

class HighResVGGEncoder(nn.Module):
    def __init__(self, embedding_size=512, input_modes=1):
        super(HighResVGGEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.convs1 = ConvSequence(input_modes,   64,  kernel_size=(3,3), padding=1)
        self.convs2 = ConvSequence(64, 128,  kernel_size=(3,3), padding=1)
        self.convs3 = ConvSequence(128, 256, kernel_size=(3,3), padding=1)
        self.convs4 = ConvSequence(256, 256, kernel_size=(3,3), padding=1)
        self.convs5 = ConvSequence(256, embedding_size, kernel_size=(3,3), padding=1)

    def forward(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1,kernel_size=2)
        x2 = self.convs2(x2)
        x3 = self.convs3(x2)
        x4 = F.max_pool2d(x3,kernel_size=2)
        x4 = self.convs4(x4)
        x5 = self.convs5(x4)
        return x5

    def forward_with_skips(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1,kernel_size=2)
        x2 = self.convs2(x2)
        x3 = self.convs3(x3)
        x3 = F.max_pool2d(x3,kernel_size=2)
        x4 = self.convs3(x4)
        x5 = self.convs4(x5)

        return x5,x4, x3, x2, x1

class VGGEncoder3d(nn.Module):
    def __init__(self, embedding_size=512, input_modes=1):
        super(VGGEncoder3d, self).__init__()
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


if __name__ == '__main__':
    sys.path.append('/users/rhydian/self-supervised-project')
    from train_SSEC import get_dataloaders

    # load models
    mri_vgg_encoder = VGGEncoder(input_modes=2)
    dxa_vgg_encoder = VGGEncoder(input_modes=2)

    train_dl, val_dl, test_dl = get_dataloaders(10,10,10,10,10,10,'/scratch/shared/beegfs/rhydian/UKBiobank', TRAINING_AUGMENTATION=False, EQUAL_RES=True, SINGLE_DXA=False, SINGLE_MRI=False )

    for idx, sample in enumerate(val_dl):
        dxa_img = sample['dxa_img']
        dxa_ses = dxa_vgg_encoder(sample['dxa_img'])
        import pdb; pdb.set_trace()
