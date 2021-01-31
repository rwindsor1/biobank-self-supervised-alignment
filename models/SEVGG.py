import sys, os, glob
import torch.nn as nn
import torch.nn.functional as F
import torch

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
    
class STN(nn.Module):
    def __init__(self, out_shape, rotation_only=False):
        super().__init__()
        if rotation_only:
            self.theta_parameters = 1
        else:
            self.theta_parameters = 3
        self.out_shape = out_shape

        self.img_encoder = nn.Sequential(
                                # layer 1
                                nn.Conv2d(1,64, kernel_size=(3,3), padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(64),
                                nn.MaxPool2d(kernel_size=2),
                                # layer 2
                                nn.Conv2d(64,128, kernel_size=(3,3), padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(128),
                                nn.MaxPool2d(kernel_size=2),
                                # layer 3
                                nn.Conv2d(128,256, kernel_size=(3,3), padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(256))

        self.localiser_fcs = nn.Sequential(
                                nn.Linear(256,512),
                                nn.ReLU(),
                                nn.Linear(512, self.theta_parameters))

    def get_theta(self, x):
        encoded_img = self.img_encoder(x)
        x = torch.max_pool2d(encoded_img,encoded_img.shape[2:]).view(encoded_img.shape[0],-1)

        theta_vals = self.localiser_fcs(x)

        if self.theta_parameters == 1:
            angles = theta_vals[:,0]; dx = torch.zeros_like(angles); dy = torch.zeros_like(angles)
        else:
            angles = theta_vals[:,0]; dx = theta_vals[:,1]; dy = theta_vals[:,2]

        c = torch.cos(angles); s = torch.sin(angles)

        theta   = torch.stack((c, -s, dx, s, c, dy), dim=-1).view(-1, 2, 3)
        return theta

    def forward(self, x):
        theta   = self.get_theta(x)
        grid    = F.affine_grid(theta, x.size(), align_corners=False)
        out_img = F.grid_sample(x, grid, align_corners=False)[:,:,:self.out_shape[0],:self.out_shape[1]]
        return out_img, theta




class SEVGG(nn.Module):
    def __init__(self, embedding_size, spatial_embeds_shape, rotation_only=False):
        super().__init__()
        self.stn = STN(spatial_embeds_shape, rotation_only=rotation_only)
        self.convs1 = ConvSequence(1,   64,  kernel_size=(3,3), padding=1)
        self.convs2 = ConvSequence(64, 128,  kernel_size=(3,3), padding=1)
        self.convs3 = ConvSequence(128, 256, kernel_size=(3,3), padding=1) 
        self.convs4 = ConvSequence(256, 512, kernel_size=(3,3), padding=1) 
        self.scan_lvl_convs = nn.Sequential(nn.Conv2d(512,512, kernel_size=(1,1)),
                                            nn.ReLU(),
                                            nn.Conv2d(512,embedding_size, kernel_size=(1,1)))


    def get_spatial_embeddings(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1,kernel_size=2)
        x2 = self.convs2(x2)
        x3 = F.max_pool2d(x2,kernel_size=2)
        x3 = self.convs3(x3)
        x4 = F.max_pool2d(x3, kernel_size=2)
        x4 = self.convs4(x4)
        return self.scan_lvl_convs(x4)

    def forward(self, x):
        xdash, theta = self.stn(x)
        embeds = self.get_spatial_embeddings(xdash)
        return embeds
    
    def transform_img(self, x):
        xdash, theta = self.stn(x)
        return xdash, theta

if __name__ == '__main__':
    sys.path.append('/users/rhydian/self-supervised-project')
    from datasets.MidCoronalDataset import MidCoronalDataset
    DATASET_ROOT = '/scratch/shared/beegfs/rhydian/UKBiobank'
    BLANKING_AUGMENTATION = False

    train_ds = MidCoronalDataset(set_type='train',
                                 all_root=DATASET_ROOT,
                                 augment=True,
                                 blanking_augment=BLANKING_AUGMENTATION)

                
    sample = train_ds[0]
    out_shape = sample['mri_img'].shape
    mri_model = SEVGG(64, (out_shape[-2], out_shape[-1]), rotation_only=False)
    dxa_model = SEVGG(64, (out_shape[-2], out_shape[-1]), rotation_only=False)
    mri_embed = F.normalize(mri_model(sample['mri_img'][None]),dim=1)
    dxa_embed = F.normalize(dxa_model(sample['dxa_img'][None]),dim=1)
    import pdb; pdb.set_trace() 


