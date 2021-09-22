import torch
import torch.nn.functional as F

class NCELoss(torch.nn.Module):
    def __init__(self, temperature=0.1)-> None:
        super().__init__()
        self.temperature = temperature
        
    def forward(self, batch_similarities):
        ax1_softmaxes = F.softmax(batch_similarities/self.temperature,dim=1)
        ax2_softmaxes = F.softmax(batch_similarities/self.temperature,dim=0)
        softmax_scores = torch.cat((-ax1_softmaxes.diag().log(),-ax2_softmaxes.diag().log()))
        loss = softmax_scores.mean()
        return loss