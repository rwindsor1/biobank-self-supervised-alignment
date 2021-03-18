import sys, os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvSequence3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvSequence3d, self).__init__()
        self.layers = nn.Sequential(nn.Conv3d(in_channels, out_channels, (kernel_size,1,kernel_size), padding=(padding,0,padding), stride=stride, bias=False),
                                    nn.BatchNorm3d(out_channels),
                                    nn.ReLU(),
                                    nn.Conv3d(out_channels, out_channels, (kernel_size, kernel_size,1), padding=(padding,padding,0), stride=1, bias=False),
                                    nn.BatchNorm3d(out_channels),
                                    nn.ReLU())

    def forward(self,x):
        return self.layers(x)

class AttentionAggregator(nn.Module):
    def __init__(self, input_embedding_size, output_embedding_size):
        super().__init__()
        self.Q = nn.Sequential(
                 nn.Conv3d(input_embedding_size, input_embedding_size, 1),
                 nn.ReLU(),
                 nn.Conv3d(input_embedding_size, 1, 1))

        self.V = nn.Sequential(
                 nn.Conv3d(input_embedding_size, input_embedding_size,1),
                 nn.ReLU(),
                 nn.Conv3d(input_embedding_size, output_embedding_size,1))
    
    def forward(self, x):
        queries = self.Q(x)
        query_vals = F.softmax(queries, dim=-2)
        values =  self.V(x)
        output = (query_vals*values).sum(dim=-2)
        return output, query_vals

class VGGEncoder3d(nn.Module):
    def __init__(self, embedding_size=512, input_modes=1, use_attention=True):
        super(VGGEncoder3d, self).__init__()
        chls = [8, 16, 32]
        self.convs1 = ConvSequence3d(input_modes,chls[0] , kernel_size=3, padding=1)
        self.convs2 = ConvSequence3d(chls[0], chls[1],  kernel_size=3, padding=1)
        self.convs3 = ConvSequence3d(chls[1], chls[2], kernel_size=3, padding=1) 
        self.convs4 = ConvSequence3d(chls[2], embedding_size, kernel_size=3, padding=1) 
        self.use_attention = use_attention
        
        if self.use_attention:
            self.attention_module = AttentionAggregator(embedding_size, embedding_size)
        else:
            self.pooling_function = lambda x: F.avg_pool3d(x,(1,x.shape[-2],1))

    def forward(self, x, return_attention_map=False):
        x1 = self.convs1(x)
        x2 = F.max_pool3d(x1,kernel_size=2)
        x2 = self.convs2(x2)
        x3 = F.max_pool3d(x2,kernel_size=2)
        x3 = self.convs3(x3)
        x4 = F.max_pool3d(x3, kernel_size=2)
        x4 = self.convs4(x4)
        if self.use_attention:
            x4, query_vals = self.attention_module(x4)
        else:
            x4 = self.pooling_function(x4) 
            query_vals = None

        if return_attention_map:
            if not self.use_attention:
                raise Exception("This model doesn't use attention!")
            return x4, query_vals
        
        else:
            return x4
