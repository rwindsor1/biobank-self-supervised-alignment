import shutil
import torch
from torch.autograd import Variable
from os import makedirs, remove
from os.path import exists, join, basename, dirname
import collections
# from lib.dataloader import default_collate
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.autograd import Variable
from torch.nn import Conv2d

def Softmax1D(x,dim):
    x_k = torch.max(x,dim)[0].unsqueeze(dim)
    x -= x_k.expand_as(x)
    exp_x = torch.exp(x)
    return torch.div(exp_x,torch.sum(exp_x,dim).unsqueeze(dim).expand_as(x))
    
def conv4d(data,filters,bias=None,permute_filters=True,use_half=False):
    b,c,h,w,d,t=data.size()

    data=data.permute(2,0,1,3,4,5).contiguous() # permute to avoid making contiguous inside loop    
        
    # Same permutation is done with filters, unless already provided with permutation
    if permute_filters:
        filters=filters.permute(2,0,1,3,4,5).contiguous() # permute to avoid making contiguous inside loop    

    c_out=filters.size(1)
    if use_half:
        output = Variable(torch.HalfTensor(h,b,c_out,w,d,t),requires_grad=data.requires_grad)
    else:
        output = Variable(torch.zeros(h,b,c_out,w,d,t),requires_grad=data.requires_grad)
    
    padding=filters.size(0)//2
    if use_half:
        Z=Variable(torch.zeros(padding,b,c,w,d,t).half())
    else:
        Z=Variable(torch.zeros(padding,b,c,w,d,t))
    
    if data.is_cuda:
        Z=Z.cuda(data.get_device())    
        output=output.cuda(data.get_device())
        
    data_padded = torch.cat((Z,data,Z),0)
    

    for i in range(output.size(0)): # loop on first feature dimension
        # convolve with center channel of filter (at position=padding)
        output[i,:,:,:,:,:]=F.conv3d(data_padded[i+padding,:,:,:,:,:], 
                                     filters[padding,:,:,:,:,:], bias=bias, stride=1, padding=padding)
        # convolve with upper/lower channels of filter (at postions [:padding] [padding+1:])
        for p in range(1,padding+1):
            output[i,:,:,:,:,:]=output[i,:,:,:,:,:]+F.conv3d(data_padded[i+padding-p,:,:,:,:,:], 
                                                             filters[padding-p,:,:,:,:,:], bias=None, stride=1, padding=padding)
            output[i,:,:,:,:,:]=output[i,:,:,:,:,:]+F.conv3d(data_padded[i+padding+p,:,:,:,:,:], 
                                                             filters[padding+p,:,:,:,:,:], bias=None, stride=1, padding=padding)

    output=output.permute(1,2,0,3,4,5).contiguous()
    return output

class Conv4d(_ConvNd):
    """Applies a 4D convolution over an input signal composed of several input
    planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, pre_permuted_filters=True): 
        # stride, dilation and groups !=1 functionality not tested 
        stride=1
        dilation=1
        groups=1
        # zero padding is added automatically in conv4d function to preserve tensor size
        padding = 0
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        super(Conv4d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _quadruple(0), groups, bias, padding_mode='zeros')  
        # weights will be sliced along one dimension during convolution loop
        # make the looping dimension to be the first one in the tensor, 
        # so that we don't need to call contiguous() inside the loop
        self.pre_permuted_filters=pre_permuted_filters
        if self.pre_permuted_filters:
            self.weight.data=self.weight.data.permute(2,0,1,3,4,5).contiguous()
        self.use_half=False


    def forward(self, input):
        return conv4d(input, self.weight, bias=self.bias,permute_filters=not self.pre_permuted_filters,use_half=self.use_half) # filters pre-permuted in constructor
def collate_custom(batch):
    """ Custom collate function for the Dataset class
     * It doesn't convert numpy arrays to stacked-tensors, but rather combines them in a list
     * This is useful for processing annotations of different sizes
    """    
    # this case will occur in first pass, and will convert a
    # list of dictionaries (returned by the threads by sampling dataset[idx])
    # to a unified dictionary of collated values
    if isinstance(batch[0], collections.Mapping):
        return {key: collate_custom([d[key] for d in batch]) for key in batch[0]}
    # these cases will occur in recursion
    elif torch.is_tensor(batch[0]): # for tensors, use standrard collating function
        return default_collate(batch)
    else: # for other types (i.e. lists), return as is
        return batch

class BatchTensorToVars(object):
    """Convert tensors in dict batch to vars
    """
    def __init__(self, use_cuda=True):
        self.use_cuda=use_cuda
        
    def __call__(self, batch):
        batch_var = {}
        for key,value in batch.items():
            if isinstance(value,torch.Tensor) and not self.use_cuda:
                batch_var[key] = Variable(value,requires_grad=False)
            elif isinstance(value,torch.Tensor) and self.use_cuda:
                batch_var[key] = Variable(value,requires_grad=False).cuda()
            else:
                batch_var[key] = value            
        return batch_var
    
def save_checkpoint(state, is_best, file, save_all_epochs=False):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir!='' and not exists(model_dir):
        makedirs(model_dir)
    if save_all_epochs:
        torch.save(state, join(model_dir,str(state['epoch'])+'_' + model_fn))
        if is_best:
            shutil.copyfile(join(model_dir,str(state['epoch'])+'_' + model_fn), join(model_dir,'best_' + model_fn))
    else:
        torch.save(state, file)
        if is_best:
            shutil.copyfile(file, join(model_dir,'best_' + model_fn))

        
def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def expand_dim(tensor,dim,desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))
        