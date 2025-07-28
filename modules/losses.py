import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiverseExpertLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = F.cross_entropy 

    def forward(self, targets, extra_info=None, base_probs=None, alpha=1.0):
 
        prior = base_probs
        prior = torch.where(prior < 0.1, 0.1, prior)

        # Obtain logits from each expert  
        expert1_logits = extra_info['logits'][0]
        expert2_logits = extra_info['logits'][1] 
        expert3_logits = extra_info['logits'][2]  

        expert1_logits[:, 3, :14] += torch.log(prior[3].view(1, -1) + 1e-9)
        expert1_logits[:, 1, :14] += alpha * torch.log(prior[1].view(1, -1) + 1e-9)
        loss_1 = self.base_loss(expert1_logits, targets)

        # loss_2_reg = torch.sum(F.softmax(expert2_logits, dim=1) * F.log_softmax(expert2_logits, dim=1), dim=-1).mean()
        expert2_logits[:, 1, :14] += torch.log(prior[1].view(1, -1) + 1e-9)
        loss_2 = self.base_loss(expert2_logits, targets)

        # loss_3_reg = torch.sum(F.softmax(expert3_logits, dim=1) * F.log_softmax(expert3_logits, dim=1), dim=-1).mean()
        expert3_logits[:, 1, :14] += alpha * torch.log(prior[1].view(1, -1) + 1e-9)
        expert3_logits[:, 2, :14] += torch.log(prior[2].view(1, -1) + 1e-9)
        loss_3 = self.base_loss(expert3_logits, targets)

        loss = loss_1 + loss_2 + loss_3
        return loss, [loss_1, loss_2, loss_3]

