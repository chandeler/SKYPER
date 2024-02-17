import torch
import torch.nn as nn
class ContrasiveLoss(nn.Module):
    def __init__(self):
        super(ContrasiveLoss, self).__init__()
        return

    def forward(self, predictions, temperature):
        prediction = torch.exp(predictions/temperature)
        positive = prediction[:,0]
        negative = torch.mean(prediction[:,1:],dim=-1).squeeze()
        loss = -torch.log(torch.mean(positive/(positive+negative)))
        return loss