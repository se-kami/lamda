#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

class Noise(nn.Module):
    def __init__(self, mean=0.0, sigma=0.1):
        super().__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            eps = torch.empty(x.shape).normal_(mean=self.mean, std=self.sigma).to(x.device)
            # eps = torch.autograd.Variable(torch.empty(x.shape).normal_(mean=self.mean, std=self.sigma).to(x.device))
            x = x + eps
        return x


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

# small
class Generator_Small(nn.Module):
    def __init__(self, in_size=2048, out_size=256):
        super().__init__()
        self.in_size = in_size[0]
        self.out_size = out_size

        generator = []
        generator.append(nn.Linear(self.in_size, self.out_size))
        generator.append(nn.BatchNorm1d(self.out_size))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.Dropout(p=0.5))
        generator.append(Noise(sigma=1))
        self.generator = nn.Sequential(*generator)

    def forward(self, x):
        x = self.generator(x)
        return x


class Classifier_Small(nn.Module):
    def __init__(self, out_size=10, in_size=256):
        super().__init__()
        self.out_size = out_size
        self.in_size = in_size

        classifier = []
        classifier.append(Flatten())
        classifier.append(nn.Linear(self.in_size, self.out_size))
        classifier.append(nn.LeakyReLU(0.1))
        classifier.append(nn.BatchNorm1d(self.out_size))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.classifier(x)
        return x

# medium
class Generator_Medium(nn.Module):
    def __init__(self, in_size=(3, 32, 32)):
        super().__init__()
        self.in_size = in_size

        generator = []
        generator.append(nn.InstanceNorm2d(self.in_size[0]))
        generator.append(nn.Conv2d(3, 64, 3, padding=1))
        generator.append(nn.BatchNorm2d(64))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.Conv2d(64, 64, 3, padding=1))
        generator.append(nn.BatchNorm2d(64))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.Conv2d(64, 64, 3, padding=1))
        generator.append(nn.BatchNorm2d(64))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.MaxPool2d(2, 2))
        generator.append(nn.Dropout(p=0.5))
        generator.append(Noise(sigma=1))
        generator.append(nn.Conv2d(64, 64, 3, padding=1))
        generator.append(nn.BatchNorm2d(64))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.Conv2d(64, 64, 3, padding=1))
        generator.append(nn.BatchNorm2d(64))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.Conv2d(64, 64, 3, padding=1))
        generator.append(nn.BatchNorm2d(64))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.MaxPool2d(2, 2))
        generator.append(nn.Dropout(p=0.5))
        generator.append(Noise(sigma=1))
        self.generator = nn.Sequential(*generator)

    def forward(self, x):
        x = self.generator(x)
        return x


class Classifier_Medium(nn.Module):
    def __init__(self, out_size=10):
        super().__init__()
        self.out_size = out_size

        classifier = []
        classifier.append(nn.Conv2d(64, 64, 3, padding=1))
        classifier.append(nn.BatchNorm2d(64))
        classifier.append(nn.LeakyReLU(0.1))
        classifier.append(nn.Conv2d(64, 64, 3, padding=1))
        classifier.append(nn.BatchNorm2d(64))
        classifier.append(nn.LeakyReLU(0.1))
        classifier.append(nn.Conv2d(64, 64, 3, padding=1))
        classifier.append(nn.BatchNorm2d(64))
        classifier.append(nn.LeakyReLU(0.1))
        classifier.append(nn.AdaptiveAvgPool2d(1))
        classifier.append(Flatten())
        classifier.append(nn.Linear(64, self.out_size))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.classifier(x)
        return x


# large
class Generator_Large(nn.Module):
    def __init__(self, in_size=(3, 32, 32)):
        super().__init__()
        self.in_size = in_size

        generator = []
        generator.append(nn.InstanceNorm2d(self.in_size[0]))
        generator.append(nn.Conv2d(3, 96, 3, padding=1))
        generator.append(nn.BatchNorm2d(96))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.Conv2d(96, 96, 3, padding=1))
        generator.append(nn.BatchNorm2d(96))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.Conv2d(96, 96, 3, padding=1))
        generator.append(nn.BatchNorm2d(96))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.MaxPool2d(2, 2))
        generator.append(nn.Dropout(p=0.5))
        generator.append(Noise(sigma=1))
        generator.append(nn.Conv2d(96, 192, 3, padding=1))
        generator.append(nn.BatchNorm2d(192))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.Conv2d(192, 192, 3, padding=1))
        generator.append(nn.BatchNorm2d(192))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.Conv2d(192, 192, 3, padding=1))
        generator.append(nn.BatchNorm2d(192))
        generator.append(nn.LeakyReLU(0.1))
        generator.append(nn.MaxPool2d(2, 2))
        generator.append(nn.Dropout(p=0.5))
        generator.append(Noise(sigma=1))
        self.generator = nn.Sequential(*generator)

    def forward(self, x):
        x = self.generator(x)
        return x


class Classifier_Large(nn.Module):
    def __init__(self, out_size=10):
        super().__init__()
        self.out_size = out_size

        classifier = []
        classifier.append(nn.Conv2d(192, 192, 3, padding=1))
        classifier.append(nn.BatchNorm2d(192))
        classifier.append(nn.LeakyReLU(0.1))
        classifier.append(nn.Conv2d(192, 192, 3, padding=1))
        classifier.append(nn.BatchNorm2d(192))
        classifier.append(nn.LeakyReLU(0.1))
        classifier.append(nn.Conv2d(192, 192, 3, padding=1))
        classifier.append(nn.BatchNorm2d(192))
        classifier.append(nn.LeakyReLU(0.1))
        classifier.append(nn.AdaptiveAvgPool2d(1))
        classifier.append(Flatten())
        classifier.append(nn.Linear(192, self.out_size))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.classifier(x)
        return x


# domain discriminators
class Discriminator_1(nn.Module):
    def __init__(self, in_size=64, out_size=10):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size

        net = []
        net.append(nn.Conv2d(self.in_size, 64, 3, padding=1))
        net.append(nn.BatchNorm2d(64))
        net.append(nn.LeakyReLU(0.1))
        net.append(nn.Conv2d(64, 64, 3, padding=1))
        net.append(nn.BatchNorm2d(64))
        net.append(nn.LeakyReLU(0.1))
        net.append(nn.Conv2d(64, 64, 3, padding=1))
        net.append(nn.BatchNorm2d(64))
        net.append(nn.LeakyReLU(0.1))
        net.append(nn.AdaptiveAvgPool2d(1))
        net.append(Flatten())
        net.append(nn.Linear(64, self.out_size + 1))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator_2(nn.Module):
    def __init__(self, in_size=256, out_size=10):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size

        net = []
        net.append(Flatten())
        net.append(nn.Linear(self.in_size, self.out_size + 1))
        # net.append(nn.ReLU())
        self.net = nn.Sequential(*net)


    def forward(self, x):
        x = self.net(x)
        return x


# decode
class Decoder(nn.Module):
    def __init__(self, in_size=256, out_size=2048):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        net = []
        net.append(nn.Linear(self.in_size, self.out_size))
        net.append(nn.BatchNorm1d(self.out_size))
        net.append(nn.LeakyReLU(0.1))
        net.append(nn.Dropout(p=0.5))
        net.append(Noise(sigma=1))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x


# lamda
class LAMDA():
    encoders = {'small': Generator_Small,
                'medium': Generator_Medium,
                'large': Generator_Large}
    classifiers = {'small': Classifier_Small,
                   'medium': Classifier_Medium,
                   'large': Classifier_Large}
    domain_discriminators = {'small': Discriminator_2,
                             'medium': Discriminator_2,
                             'large': Discriminator_1}
    mid_sizes = {'small': 256,
                 'medium': 64,
                 'large': 192}

    def __init__(self, model_size, in_size, out_size,
                 share_encoders=True, decay=0.998):
        """
        model_size: 'small', 'medium' or 'large'
        in_size: int or tuple
        out_size: int
        share_encoders: share encoders
        decay: used for EMA
        """
        self.device = 'cpu'
        self.model_size = model_size.lower()
        self.share_encoders = share_encoders
        self.decay = decay

        mid_size = LAMDA.mid_sizes[self.model_size]

        if self.model_size == 'small':
            self.encoder_source = LAMDA.encoders[self.model_size](in_size, mid_size)
            if self.share_encoders:
                self.encoder_target = self.encoder_source
            else:
                self.encoder_target = LAMDA.encoders[self.model_size](in_size, mid_size)
            self.class_disc = LAMDA.classifiers[self.model_size](out_size, mid_size)
            self.domain_disc = LAMDA.domain_discriminators[self.model_size](mid_size, out_size)
        else:
            self.encoder_source = LAMDA.encoders[self.model_size](in_size)
            if self.share_encoders:
                self.encoder_target = self.encoder_source
            else:
                self.encoder_target = LAMDA.encoders[self.model_size](in_size)
            self.class_disc = LAMDA.classifiers[self.model_size](out_size)
            self.domain_disc = LAMDA.domain_discriminators[self.model_size](mid_size, out_size)

        # decoder
        if type(in_size) == int:
            self.decoder = Decoder(mid_size, in_size)
        else:
            decoder_out_size = 1
            for i in in_size:
                decoder_out_size *= i
            self.decoder = Decoder(mid_size, decoder_out_size)

    def save_models(self, name='model.pth'):
        dicts = {
                'encoder_source': self.encoder_source.state_dict(),
                'encoder_target': self.encoder_target.state_dict(),
                'class_disc': self.class_disc.state_dict(),
                'domain_disc': self.domain_disc.state_dict(),
                'decoder': self.decoder.state_dict(),
                }

        torch.save(dicts, name)

    def load_models(self, path='model.pth'):
        dicts = torch.load(path)

        self.encoder_source.load_state_dict(dicts['encoder_source'])
        self.encoder_target.load_state_dict(dicts['encoder_target'])
        self.class_disc.load_state_dict(dicts['class_disc'])
        self.domain_disc.load_state_dict(dicts['domain_disc'])
        self.decoder.load_state_dict(dicts['decoder'])

    def load_lamda(self, lamda):
        """
        copies weight of another LAMDA model
        """
        self.model_size = lamda.model_size
        self.share_encoders = lamda.share_encoders

        self.encoder_source.load_state_dict(lamda.encoder_source.state_dict())
        if self.share_encoders:
            self.encoder_target = self.encoder_source
        else:
            self.encoder_target.load_state_dict(lamda.encoder_target.state_dict())

        self.class_disc.load_state_dict(lamda.class_disc.state_dict())
        self.domain_disc.load_state_dict(lamda.domain_disc.state_dict())
        self.decoder.load_state_dict(lamda.decoder.state_dict())

    def train(self):
        """
        set to train mode
        """
        self.encoder_source.train()
        self.encoder_target.train()
        self.class_disc.train()
        self.domain_disc.train()
        self.decoder.train()

    def eval(self):
        """
        set to eval mode
        """
        self.encoder_source.eval()
        self.encoder_target.eval()
        self.class_disc.eval()
        self.domain_disc.eval()
        self.decoder.eval()

    def to(self, device):
        """
        move to device
        """
        self.device = device
        self.encoder_source.to(device)
        self.encoder_target.to(device)
        self.class_disc.to(device)
        self.domain_disc.to(device)
        self.decoder.to(device)
        return self

    def get_primary_params(self):
        params = [
                {'params': self.encoder_source.parameters()},
                {'params': self.class_disc.parameters()},
                {'params': self.decoder.parameters()},
                ]
        if not self.share_encoders:
            params.append({'params': self.encoder_target.parameters()})
        return params

    def get_domain_disc_params(self):
        return self.domain_disc.parameters()

    @torch.no_grad()
    def ema_update(self, model):
        if self.decay <= 0:
            return
        ema_nets = ['encoder_source', 'class_disc', 'domain_disc', 'decoder']
        if not self.share_encoders:
            ema_nets.append('encoder_target')

        # ema update
        for net in ema_nets:
            this_model = getattr(self, net)
            that_model = getattr(model, net)
            for (this_name, this_param), (that_name, that_param) in zip(
                    list(this_model.named_parameters()),
                    list(that_model.named_parameters())):
                this_param -= (1 - self.decay) * (this_param.data - that_param.data)

    def get_device(self):
        device = next(self.encoder_source.parameters()).device
        return device

    def __call__(self, x):
        x_mid = self.encoder_source(x)
        logit = self.class_disc(x_mid)
        return logit

    @torch.no_grad()
    def predict_domain(self, x):
        x_mid = self.encoder_source(x)
        pred = torch.sigmoid(self.domain_disc(x_mid)[:, -1])
        return pred

    @torch.no_grad()
    def embed(self, x):
        x_mid = self.encoder_source(x)
        return x_mid
