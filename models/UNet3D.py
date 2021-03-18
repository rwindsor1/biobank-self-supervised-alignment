import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvSequence(nn.Module):
    def __init__(self,in_c,mid_c,out_c , kernel_size,padding,out_bnrm_and_relu=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c,mid_c,(kernel_size,kernel_size,kernel_size),padding=padding)
        self.bnrm1 = nn.BatchNorm3d(mid_c)
        self.conv2 = nn.Conv3d(mid_c,out_c,(kernel_size,kernel_size,kernel_size),padding=padding)
        if out_bnrm_and_relu:
            self.bnrm2 = nn.BatchNorm3d(out_c)
            self.out_bnrm_and_relu=True
        else:
            self.bnrm2 = None
            self.out_bnrm_and_relu=False

    def forward(self,x):
        x = self.conv2(F.relu(self.bnrm1(self.conv1(x))))
        if self.out_bnrm_and_relu:
            x = F.relu(self.bnrm2(x))
        return x

class UNet3D(nn.Module):
    def __init__(self,in_dim, out_dim):
        super().__init__()
        c = [in_dim,8,16,32,64,128]
        self.down_conv1 = ConvSequence(c[0],c[1],c[2],3,1)
        self.down_conv2 = ConvSequence(c[2],c[2],c[3],3,1)
        self.down_conv3 = ConvSequence(c[3],c[3],c[4],3,1)
        self.mid_conv = ConvSequence(c[4],c[4],c[5],3,1)
        self.up_conv1 = ConvSequence(c[5]+c[4],c[4],c[4],3,1)
        self.up_conv2 = ConvSequence(c[4]+c[3],c[3],c[3],3,1)
        self.up_conv3 = ConvSequence(c[3]+c[2],c[2],out_dim,3,1,False)
        self.c = c

    def forward(self,x_in):
        x1 = self.down_conv1(x_in)
        x2 = self.down_conv2(F.max_pool3d(x1,2))
        x3 = self.down_conv3(F.max_pool3d(x2,2))
        out = self.mid_conv(F.max_pool3d(x3,2))
        out = torch.cat([x3,F.interpolate(out, x3.shape[-3:])],dim=1)
        out = torch.cat([x2,F.interpolate(self.up_conv1(out), x2.shape[-3:])],dim=1)
        out = torch.cat([x1,F.interpolate(self.up_conv2(out), x1.shape[-3:])],dim=1)
        out = self.up_conv3(out)
        return out


if __name__ == '__main__':
    import os, pickle
    array_path = '/scratch/shared/beegfs/rhydian/UKBiobank/Cleaned3DMRIsHalfPrecision/train'
    array_name = os.listdir(array_path)[0]
    model = UNet3D(1,1)
    with open(os.path.join(array_path,array_name),'rb') as f:
        scan = pickle.load(f)

    model(scan.float()[None,None])

    import pdb; pdb.set_trace()
