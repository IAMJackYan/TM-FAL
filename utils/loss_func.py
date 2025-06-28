import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn
import pdb
import logging

# class Dice_Loss(nn.Module):
#     def __init__(self):
#         super(Dice_Loss, self).__init__()
#         self.smooth = 1e-5

#     def forward(self, pred, label):
#         # one_hot_y = torch.zeros(pred.shape).cuda()
#         # one_hot_y = one_hot_y.scatter_(1, label, 1.0, reduce='multiply')
#         # one_hot_y.requires_grad = False

#         # K = pred.shape[1]
#         # one_hot_y = F.one_hot(label[:,0,:,:], num_classes=K)
#         # one_hot_y = one_hot_y.permute(0, 3, 1, 2).float()
#         # one_hot_y.requires_grad = False
#         one_hot_y = label

#         dice_score = 0.0
        
#         for class_idx in range(pred.shape[1]):   # 正类  
#             inter = (pred[:,class_idx,...] * one_hot_y[:,class_idx,...]).sum()
#             union = (pred[:,class_idx,...] ** 2).sum() + (one_hot_y[:,class_idx,...] ** 2).sum()

#             dice_score += (2*inter + self.smooth) / (union + self.smooth)

#         loss = 1 - dice_score/pred.shape[1]

#         return loss

class Dice_Loss(nn.Module):
    def __init__(self):
        super(Dice_Loss, self).__init__()
        self.smooth = 1e-5

    def forward(self, pred, label):
        one_hot_y = torch.zeros(pred.shape).cuda()
        one_hot_y = one_hot_y.scatter_(1, label, 1.0)

        dice_score = 0.0
        
        for class_idx in range(pred.shape[1]):   # 正类  
            inter = (pred[:,class_idx,...] * one_hot_y[:,class_idx,...]).sum()
            union = (pred[:,class_idx,...] ** 2).sum() + (one_hot_y[:,class_idx,...] ** 2).sum()

            dice_score += (2*inter + self.smooth) / (union + self.smooth)

        loss = 1 - dice_score/pred.shape[1]

        return loss

def kl_divergence(alpha):
    shape = list(alpha.shape)
    shape[0] = 1
    ones = torch.ones(tuple(shape)).cuda()

    S = torch.sum(alpha, dim=1, keepdim=True) 
    first_term = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(S))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl.mean()    # / batch_size


class EDL_Loss(nn.Module):
    def __init__(self, prior, kl_weight=0.01, annealing_step=10):
        super(EDL_Loss, self).__init__()
        self.prior = prior
        self.kl_weight = kl_weight
        self.annealing_step = annealing_step

    def forward(self, logit, label, epoch_num):
        K = logit.shape[1]
        alpha = F.relu(logit) + 1
        S = torch.sum(alpha, dim=1, keepdim=True) 

        one_hot_y = torch.eye(K).cuda()
        one_hot_y = one_hot_y[label]
        one_hot_y.requires_grad = False

        loss_ce = torch.sum((2-self.prior) * one_hot_y * (torch.digamma(S) - torch.digamma(alpha))) / logit.shape[0]

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
        ) 

        kl_alpha = (alpha - 1) * (1 - one_hot_y) + 1
        loss_kl = annealing_coef * kl_divergence(kl_alpha)

        loss_cor = torch.sum(- K/S.detach() * one_hot_y * logit) / logit.shape[0]      
        # print('ce loss: {}, kl loss: {}, cor loss: {}'.format(loss_ce.item(), loss_kl.item(), loss_cor.item()))

        return loss_ce + self.kl_weight * (loss_kl+loss_cor)
    

class EDL_Dice_Loss(nn.Module):
    def __init__(self, kl_weight=0.001, annealing_step=10):
        super(EDL_Dice_Loss, self).__init__()
        self.smooth = 1e-5
        self.kl_weight = kl_weight
        self.annealing_step = annealing_step

    def forward(self, logit, label, epoch_num):
        K = logit.shape[1]

        alpha = (F.relu(logit)+1)**2
        S = torch.sum(alpha, dim=1, keepdim=True) 

        pred = alpha / S

        one_hot_y = torch.zeros(pred.shape).cuda()
        one_hot_y = one_hot_y.scatter_(1, label, 1.0)
        one_hot_y.requires_grad = False

        dice_score = 0
        for class_idx in range(logit.shape[1]):   
            inter = (pred[:,class_idx,...] * one_hot_y[:,class_idx,...]).sum()
            union = (pred[:,class_idx,...] ** 2).sum() + (one_hot_y[:,class_idx,...] ** 2).sum() + (pred[:,class_idx,...]*(1-pred[:,class_idx,...])/(S[:,0,...]+1)).sum()

            dice_score += (2*inter + self.smooth) / (union + self.smooth)

        loss_dice = 1 - dice_score/logit.shape[1]

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
        ) 

        kl_alpha = (alpha - 1) * (1 - one_hot_y) + 1
        loss_kl = annealing_coef * kl_divergence(kl_alpha)

        loss_cor = torch.sum(- K/S.detach() * one_hot_y * logit) / (logit.shape[0] * logit.shape[2] * logit.shape[3])
        # print('dice loss: {}, kl loss: {}, cor loss: {}'.format(loss_dice.item(), loss_kl.item(), loss_cor.item()))

        return loss_dice + self.kl_weight * (loss_kl+loss_cor)



def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5

    loss = 0
    for i in range(target.shape[1]):
        intersect = torch.sum(score[:, i, ...] * target[:, i, ...])
        z_sum = torch.sum(score[:, i, ...] )
        y_sum = torch.sum(target[:, i, ...] )
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss * 1.0 / target.shape[1]

    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)