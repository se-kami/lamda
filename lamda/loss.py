#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F

class CondEntropyLoss(nn.Module):
    """
    loss between two logits
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logit, target):
        target_p = torch.softmax(target, 1)
        logit_p = torch.log_softmax(logit, 1)
        x = -torch.sum(logit_p * target_p, dim=1)
        if self.reduction == 'mean':
            x = x.mean(dim=0)
        return x


class VATLoss(nn.Module):
    """
    virtual adversarial loss
    """
    def __init__(self, classifier, generator):
        super().__init__()
        self.classifier = classifier
        self.generator = generator

    def forward(self, x, p):
        x_adv = perturb_image(x, p, self.classifier, self.generator)
        x_adv_mid = self.generator(x_adv)
        x_adv_pred = self.classifier(x_adv_mid)
        # loss = cross_entropy_with_mean(x_adv_pred, torch.softmax(p.detach()))
        # loss =  -torch.sum(torch.log_softmax(x_adv_pred, dim=1) * torch.softmax(p.detach(), dim=1), dim=1).mean(dim=0)
        loss = CondEntropyLoss(reduction='mean')(x_adv_pred, p.detach())
        return loss


def perturb_image(x, p, classifier, encoder, radius=3.5, power=1):
    d = torch.rand_like(x)
    # d = 1e-6 * F.normalize(d.reshape(d.size(0), -1), p=2, dim=1).requires_grad_()
    d = d.reshape(d.size(0), -1)
    d = 1e-6 * F.normalize(d, p=2, dim=1).reshape(x.shape).requires_grad_()
    x_eps_mid = encoder(x+d)
    x_eps_pred = classifier(x_eps_mid)
    loss = CondEntropyLoss(reduction='mean')(x_eps_pred, p)
    grad = torch.autograd.grad(loss, [d])[0]
    eps_adv = radius * F.normalize(grad.reshape(grad.size(0), -1), p=2, dim=1).reshape(x.shape)
    eps_adv = x + eps_adv * radius
    eps_adv = eps_adv.detach()
    return eps_adv


class ReconstructionLoss(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder


    def forward(self, x_src, x_mid):
        x_decoded = self.decoder(x_mid)
        x_reconstruct = F.normalize(x_src - x_decoded, p=2, dim=1)
        loss = (x_reconstruct * x_reconstruct).mean() / 1000.0
        return loss
