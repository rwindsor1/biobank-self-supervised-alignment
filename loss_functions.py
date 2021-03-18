import torch
import torch.nn.functional as F
import numpy as np



class ContrastiveLoss(torch.nn.Module):
    def __init__(self, batch_size, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

    def forward(self, zis, zjs):
        delta_norms = (zis.unsqueeze(0)-zjs.unsqueeze(1)).norm(dim=-1)
        positives = torch.diag(delta_norms)
        negatives = delta_norms[self.mask_samples_from_same_repr].view(self.batch_size-1, self.batch_size)
        positives_loss = torch.sum(positives)
        negatives_loss = torch.clamp(self.margin - negatives, min=0).sum()/(self.batch_size-1)
        return positives_loss + negatives_loss

    def _get_correlated_mask(self):
        diag = np.eye(self.batch_size)
        mask = torch.from_numpy(diag)
        mask = (1 - mask).type(torch.bool)
        return mask.cuda()



def pairwise_out_loss(outs):
    true_preds = torch.sigmoid(torch.diag(outs))
    false_preds = torch.sigmoid(torch.diag(outs, 1))
    loss = F.binary_cross_entropy(true_preds, torch.ones(true_preds.shape[0]).cuda()) + F.binary_cross_entropy(false_preds, torch.zeros(false_preds.shape[0]).cuda())
    return loss


class SSECLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(SSECLoss, self).__init__()
        self.temperature = temperature

    def forward(self,similarities):
        ax1_softmaxes = F.softmax(similarities/self.temperature,dim=1)
        ax2_softmaxes = F.softmax(similarities/self.temperature,dim=0)
        softmax_scores = torch.cat((-ax1_softmaxes.diag().log(),-ax2_softmaxes.diag().log()))
        loss = softmax_scores.mean()
        return loss


if __name__ == '__main__':
    #criterion = NTXentLoss(5,1,True)
    criterion = ContrastiveLoss(5,1)
    val1 = torch.randn((5,2))
    val2 = torch.randn((5,2))
    criterion(val1,val2)

