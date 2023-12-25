import logging
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

# def r_prior(inputs):
#     # inputs shape batch, top_k, c, h, w
#     total_prior = []
#     for batch in inputs:
#         total_prior.append(_r_prior(batch).unsqeeuze(0))
#     total_prior = torch.cat(total_prior, dim=0)
#     return total_prior
    
def r_prior(inputs):
    # COMPUTE total variation regularization loss
    diff1 = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
    diff2 = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
    diff3 = inputs[:, :, 1:, :-1] - inputs[:, :, :-1, 1:]
    diff4 = inputs[:, :, :-1, :-1] - inputs[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2

# def r_feature(r_feature_layers):
#     ret = sum([mod.r_feature for (idx, mod) in enumerate(r_feature_layers)])
#     return ret
# def r_l2(inputs):
#     # inputs shape batch, top_k, c, h, w
#     total_l2 = []
#     for batch in inputs:
#         total_l2.append(_r_l2(batch))
#     total_l2 = torch.cat(total_l2, dim=0)

def r_l2(inputs):
    norm = torch.norm(inputs)
    return norm