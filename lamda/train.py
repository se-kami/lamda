#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import os
from time import time
import torch
import pandas as pd
from torch import nn
from data import get_loaders
from model import LAMDA
from device import get_device
from loss import CondEntropyLoss, VATLoss, ReconstructionLoss
from torch.utils.tensorboard import SummaryWriter


def accuracy_loader(model, loader):
    model.eval()
    device = model.get_device()
    correct, total = 0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yp = model(xb)
            yp = torch.argmax(yp, 1)
            correct += torch.sum((yp == yb).to(float)).item()
            total += len(xb)

    return correct / total


def accuracy_domain_loader(model, loader, target_domain=0):
    model.eval()
    device = model.get_device()
    correct, total = 0, 0

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            yp = model.predict_domain(xb)
            correct += torch.sum((torch.abs(yp - target_domain) < 0.5).to(float)).item()
            total += len(xb)

    return correct / total


def train(config, print_every=100, save_every=500):
    """
    print_every: how often to calculate and print out accuracies
    save_every: how often to save trained models
    """
    # make tensorboard dir
    name = config['save_dir'].split('/')[-1]
    dir_name = f'tensorboard/{name}'
    os.makedirs(f'tensorboard/{name}', exist_ok=True)
    writer = SummaryWriter(log_dir=dir_name)
    config['writer'] = writer

    # save config
    with open(config['save_dir'] + '/config.json', 'w') as f:
        config_to_save = dict()
        keys_to_ignore = ['device',
                          'model',
                          'model_ema',
                          'optimizer_primary',
                          'optimizer_domain_disc',
                          'writer',
                          'loaders',
                          'loader_train']
        for key, val in config.items():
            if key in keys_to_ignore:
                continue
            config_to_save[key] = val
        json.dump(config_to_save, f)

    # training config
    # loaders
    loaders = config['loaders']  # loaders for eval
    loader_train = config['loader_train']  # loader for train
    num_classes = config['num_classes']
    num_iters = config['num_iters']
    optimizer_primary = config['optimizer_primary']
    optimizer_domain_disc = config['optimizer_domain_disc']
    update_target_loss = config['update_target_loss']

    # load model
    device = config['device']
    lamda = config['model']
    lamda.to(device)
    lamda_ema = config['model_ema']
    lamda_ema.to(device)
    lamda_ema.eval()

    # tradeoffs
    m_plus_1_on_D_trade_off = config['m_plus_1_on_D_trade_off']
    m_on_D_trade_off = config['m_on_D_trade_off']
    src_class_trade_off = config['src_class_trade_off']
    domain_trade_off  = config['domain_trade_off']
    src_vat_trade_off = config['src_vat_trade_off']
    trg_trade_off = config['trg_trade_off']
    m_plus_1_on_G_trade_off = config['m_plus_1_on_G_trade_off']
    m_on_G_trade_off = config['m_on_G_trade_off']
    src_recons_trade_off = config['src_recons_trade_off']

    # losses
    cross_entropy_with_mean = nn.CrossEntropyLoss(reduction='mean').to(device)
    bce_with_mean = nn.BCEWithLogitsLoss(reduction='mean').to(device)
    cond_entropy_with_mean = CondEntropyLoss(reduction='mean').to(device)
    vat_loss_src = VATLoss(lamda.class_disc, lamda.encoder_source)
    vat_loss_trg = VATLoss(lamda.class_disc, lamda.encoder_target)
    reconstruct_loss = ReconstructionLoss(lamda.decoder)

    # log training settings
    writer = config['writer']
    writer.add_scalar('learning_rate', config['learning_rate'], 0)
    tradeoffs = [
            'm_plus_1_on_D_trade_off',
            'm_on_D_trade_off',
            'src_class_trade_off',
            'domain_trade_off',
            'src_vat_trade_off',
            'trg_trade_off',
            'm_plus_1_on_G_trade_off',
            'm_on_G_trade_off',
            'src_recons_trade_off',]
    for trade_off in tradeoffs:
        writer.add_scalar(trade_off, config[trade_off], 0)


    losses_batch = {
            'num_iter': [],
            'loss_m_src_on_D': [],
            'loss_m_plus_1_on_D': [],
            'loss_disc': [],
            'src_loss_class': [],
            'loss_generator': [],
            'src_loss_vat': [],
            'trg_loss_vat': [],
            'trg_loss_cond_entropy': [],
            'src_reconstruct_loss': [],
            'loss_m_trg_on_G': [],
            'loss_m_plus_1_on_G': [],
            'primary_loss': [],
            }
    if update_target_loss:
        losses_batch['target_loss'] = []

    # accuracies
    accs_batch = {
            'num_iter': [],
            'acc_batch_src': [],
            'acc_batch_trg': [],
            'acc_batch_domain_src': [],
            'acc_batch_domain_trg': [],
            }
    accs_total = {
            'num_iter': [],
            'acc_train_src': [],
            'acc_train_src_domain': [],
            'acc_train_src_ema': [],
            'acc_train_src_ema_domain': [],
            'acc_train_trg': [],
            'acc_train_trg_domain': [],
            'acc_train_trg_ema': [],
            'acc_train_trg_ema_domain': [],
            'acc_test_src': [],
            'acc_test_src_domain': [],
            'acc_test_src_ema': [],
            'acc_test_src_ema_domain': [],
            'acc_test_trg': [],
            'acc_test_trg_domain': [],
            'acc_test_trg_ema': [],
            'acc_test_trg_ema_domain': [],
            }

    # training loop
    num_iter = 0
    best_trg_acc = -1

    while num_iter < num_iters:
        num_iter += 1
        # START TRAINING
        lamda.train()

        (x_src, y_src), (x_trg, y_trg) = loader_train()
        # to device
        x_src = x_src.to(device)
        y_src = y_src.to(device)
        x_trg = x_trg.to(device)
        y_trg = y_trg.to(device)


        ### ### ### ### ### ### ###
        # optimize domain discriminator
        optimizer_domain_disc.zero_grad()
        # encode
        x_src_mid = lamda.encoder_source(x_src)
        x_trg_mid = lamda.encoder_target(x_trg)
        # domain discriminator
        x_fr_src = lamda.domain_disc(x_src_mid)
        x_fr_trg = lamda.domain_disc(x_trg_mid)
        # domain probability
        m_plus_1_src_logit_on_D = x_fr_src[:, -1]
        m_plus_1_trg_logit_on_D = x_fr_trg[:, -1]
        # classes
        m_src_on_D_logit = x_fr_src[:, :num_classes]
        # loss
        zeros = torch.zeros_like(m_plus_1_src_logit_on_D)
        src_loss = bce_with_mean(m_plus_1_src_logit_on_D, zeros)

        ones = torch.ones_like(m_plus_1_trg_logit_on_D)
        trg_loss = bce_with_mean(m_plus_1_trg_logit_on_D, ones)

        loss_m_plus_1_on_D = 0.5 * src_loss + 0.5 * trg_loss

        loss_m_src_on_D = cross_entropy_with_mean(m_src_on_D_logit, y_src)

        loss_disc = \
            m_on_D_trade_off * loss_m_src_on_D + \
            m_plus_1_on_D_trade_off * loss_m_plus_1_on_D

        # backprop
        loss_disc.backward()
        optimizer_domain_disc.step()
        # log losses
        losses_batch['num_iter'].append(num_iter)
        losses_batch['loss_m_src_on_D'].append(loss_m_src_on_D.item())
        losses_batch['loss_m_plus_1_on_D'].append(loss_m_plus_1_on_D.item())
        losses_batch['loss_disc'].append(loss_disc.item())
        # log accuracies
        accs_batch['num_iter'].append(num_iter)
        acc_batch_domain_src = (m_plus_1_src_logit_on_D <= 0.5).to(float).mean()
        accs_batch['acc_batch_domain_src'].append(acc_batch_domain_src.item())
        acc_batch_domain_trg = (m_plus_1_trg_logit_on_D > 0.5).to(float).mean()
        accs_batch['acc_batch_domain_trg'].append(acc_batch_domain_trg.item())
        ### ### ### ### ### ### ###

        ### ### ### ### ### ### ###
        # optimize generators and classifier
        optimizer_primary.zero_grad()
        # encode
        x_src_mid = lamda.encoder_source(x_src)
        x_trg_mid = lamda.encoder_target(x_trg)
        # domain discriminator
        x_fr_src = lamda.domain_disc(x_src_mid)
        x_fr_trg = lamda.domain_disc(x_trg_mid)
        # domain probability
        m_plus_1_src_logit_on_D = x_fr_src[:, -1]
        m_plus_1_trg_logit_on_D = x_fr_trg[:, -1]
        # classifier predictions
        y_src_logit = lamda.class_disc(x_src_mid)
        y_trg_logit = lamda.class_disc(x_trg_mid)

        y_src_pred = torch.argmax(y_src_logit, 1)
        y_trg_pred = torch.argmax(y_trg_logit, 1)

        # classifier loss
        src_loss_class = cross_entropy_with_mean(y_src_logit, y_src)

        # generator loss
        # loss_m_plus_1_on_G
        ones = torch.ones_like(m_plus_1_src_logit_on_D)
        src_loss = bce_with_mean(m_plus_1_src_logit_on_D, ones)

        zeros = torch.zeros_like(m_plus_1_trg_logit_on_D)
        trg_loss = bce_with_mean(m_plus_1_trg_logit_on_D, zeros)

        loss_m_plus_1_on_G = 0.5 * src_loss + 0.5 * trg_loss
        # loss_m_trg_on_G
        m_trg_on_D_logit = x_fr_trg[:, :num_classes]
        loss_m_trg_on_G = cond_entropy_with_mean(m_trg_on_D_logit, y_trg_logit)
        loss_generator = m_plus_1_on_G_trade_off * loss_m_plus_1_on_G + \
                m_on_G_trade_off * loss_m_trg_on_G

        # vat loss
        src_loss_vat = vat_loss_src(x_src, y_src_logit)
        trg_loss_vat = vat_loss_trg(x_trg, y_trg_logit)

        # target cross entropy loss
        trg_loss_cond_entropy = cond_entropy_with_mean(y_trg_logit, y_trg_logit)

        # reconstruction loss
        if src_recons_trade_off != 0.0:
            src_reconstruct_loss = reconstruct_loss(x_src, x_src_mid)
        else:
            src_reconstruct_loss = torch.tensor(0.0)

        # loss
        primary_loss = \
            src_class_trade_off * src_loss_class + \
            domain_trade_off * loss_generator + \
            src_vat_trade_off * src_loss_vat + \
            trg_trade_off * trg_loss_vat + \
            trg_trade_off * trg_loss_cond_entropy + \
            src_recons_trade_off * src_reconstruct_loss

        primary_loss.backward()
        optimizer_primary.step()

        # log losses
        losses_batch['primary_loss'].append(primary_loss.item())
        losses_batch["src_loss_class"].append(src_loss_class.item())
        losses_batch["loss_generator"].append(loss_generator.item())
        losses_batch["src_loss_vat"].append(src_loss_vat.item())
        losses_batch["trg_loss_vat"].append(trg_loss_vat.item())
        losses_batch["trg_loss_cond_entropy"].append(trg_loss_cond_entropy.item())
        losses_batch["src_reconstruct_loss"].append(src_reconstruct_loss.item())
        losses_batch["loss_m_trg_on_G"].append(loss_m_trg_on_G.item())
        losses_batch["loss_m_plus_1_on_G"].append(loss_m_plus_1_on_G.item())
        # log accuracies
        acc_batch_src = (y_src_pred == y_src).to(float).mean()
        acc_batch_trg = (y_trg_pred == y_trg).to(float).mean()
        accs_batch['acc_batch_src'].append(acc_batch_src.item())
        accs_batch['acc_batch_trg'].append(acc_batch_trg.item())

        if update_target_loss:
            x_trg_mid = lamda.encoder_target(x_trg)
            y_trg_logit = lamda.class_disc(x_trg_mid)
            trg_loss_vat = vat_loss_trg(x_trg, y_trg_logit)
            trg_loss_cond_entropy = cond_entropy_with_mean(y_trg_logit, y_trg_logit)
            target_loss = trg_trade_off * (trg_loss_vat + trg_loss_cond_entropy)
            optimizer_primary.zero_grad()
            target_loss.backward()
            optimizer_primary.step()
            losses_batch['target_loss'] = []

        # ema update
        lamda_ema.ema_update(lamda)

        # save logs
        for loss in losses_batch:
            writer.add_scalar(loss, losses_batch[loss][-1], num_iter)

        for acc in accs_batch:
            writer.add_scalar(acc, accs_batch[acc][-1], num_iter)

        # save model
        if num_iter % save_every == 0:
            name = config['save_dir'] + f'/lamda-{num_iter:06d}.pth'
            lamda.save_models(name=name)

            name_ema = config['save_dir'] + f'/lamda_ema-{num_iter:06d}.pth'
            lamda_ema.save_models(name=name_ema)

        # END TRAINING

        # START EVAL
        if num_iter == 1 or num_iter % print_every == 0 or num_iter == num_iters-1:
            lamda.eval()
            accs_total['num_iter'].append(num_iter)
            with torch.no_grad():
                prefix = f"[{num_iter:05d}/{num_iters:05d}]"

                # log accuracies
                for name, loader in loaders.items():
                    if 'src' in name:
                        target = 0.0
                    elif 'trg' in name:
                        target = 1.0

                    name = 'acc_' + name

                    acc = accuracy_loader(lamda, loader)
                    accs_total[name].append(acc)
                    acc_domain = accuracy_domain_loader(lamda, loader, target)
                    accs_total[name + '_domain'].append(acc_domain)

                    # ema
                    acc = accuracy_loader(lamda_ema, loader)
                    accs_total[name + '_ema'].append(acc)
                    acc_domain = accuracy_domain_loader(lamda_ema, loader, target)
                    accs_total[name + '_ema_domain'].append(acc_domain)

            # print
            acc_train_src = accs_total['acc_train_src'][-1] * 100
            acc_train_trg = accs_total['acc_train_trg'][-1] * 100
            acc_test_src = accs_total['acc_test_src'][-1] * 100
            acc_test_trg = accs_total['acc_test_trg'][-1] * 100
            acc_test_src_ema = accs_total['acc_test_src_ema'][-1] * 100
            acc_test_trg_ema = accs_total['acc_test_trg_ema'][-1] * 100

            print(prefix)
            print(f"acc_train_src: {acc_train_src:.2f} %", end=' ')
            print(f"acc_train_trg: {acc_train_trg:.2f} %", end='\n')
            print(f"acc_test_src: {acc_test_src:.2f} %", end=' ')
            print(f"acc_test_trg: {acc_test_trg:.2f} %", end='\n')
            print(f"acc_test_src_ema: {acc_test_src_ema:.2f} %", end=' ')
            print(f"acc_test_trg_ema: {acc_test_trg_ema:.2f} %", end='\n')

            trg_acc = accs_total['acc_test_trg'][-1] * 100
            if trg_acc > best_trg_acc:
                best_trg_acc = trg_acc
                name = config['save_dir'] + f'/lamda-best{num_iter:06d}.pth'
                lamda.save_models(name=name)

            trg_acc = accs_total['acc_test_trg_ema'][-1] * 100
            if trg_acc > best_trg_acc:
                best_trg_acc = trg_acc
                name = config['save_dir'] + f'/lamda-best-ema-{num_iter:06d}.pth'
                lamda_ema.save_models(name=name)

            for acc in accs_total:
                writer.add_scalar(acc, accs_total[acc][-1], num_iter)

            print(f"Best target accuracy {best_trg_acc:.2f} %")
            print()

    print(f"Best target accuracy {best_trg_acc:.2f} %")

    # save logs to csv
    name = config['save_dir'] + '/losses_batch.csv'
    losses_batch_df = pd.DataFrame(losses_batch)
    losses_batch_df.to_csv(name)

    name = config['save_dir'] + '/accs_batch.csv'
    accs_batch_df = pd.DataFrame(accs_batch)
    accs_batch_df.to_csv(name)

    name = config['save_dir'] + '/accs_total.csv'
    accs_total_df = pd.DataFrame(accs_total)
    accs_total_df.to_csv(name)


if __name__ == '__main__':
    default_config = {
            'm_plus_1_on_D_trade_off': 1.0,
            'm_on_D_trade_off': 1.0,
            'src_class_trade_off': 1.0,
            'domain_trade_off': 0.1,
            'src_vat_trade_off': 0.1,
            'trg_trade_off': 0.1,
            'm_plus_1_on_G_trade_off': 1.0,
            'm_on_G_trade_off': 0.1,
            'src_recons_trade_off': 0.1,
            }

    if len(sys.argv) > 1:
        file = sys.argv[1]
        with open(file, 'r') as f:
           data = json.load(f)
        default_config.update(data)
    else:
        print("Provide a config.json file.")
        sys.exit(0)
    config = default_config
    config['device'] = get_device()
    loader_train, single_loaders = get_loaders(
            source_train=config['source_train'],
            source_test=config['source_test'],
            target_train=config['target_train'],
            target_test=config['target_test'],
            batch_size_train=config['batch_size'],
            batch_size_test=config['batch_size'],
            shuffle_train=True, shuffle_test=False,
            data_dir=config['data_dir'])

    config['loader_train'] = loader_train
    config['loaders'] = {}
    config['loaders']['train_src'] = single_loaders[0]
    config['loaders']['train_trg'] = single_loaders[1]
    config['loaders']['test_src'] = single_loaders[2]
    config['loaders']['test_trg'] = single_loaders[3]

    num_classes = config['loader_train'].get_number_of_classes()
    config['num_classes'] = num_classes

    in_size = loader_train.get_in_size()  # 2048 or (3, 32, 32)
    lamda = LAMDA(model_size=config['model_size'],
                  in_size=in_size, out_size=num_classes,
                  share_encoders=config['share_encoders'])
    # ema model
    lamda_ema = LAMDA(model_size=config['model_size'],
                      in_size=in_size, out_size=num_classes,
                      share_encoders=config['share_encoders'],
                      decay=config['ema_decay'],
                      )

    lamda_ema.load_lamda(lamda)

    config['model'] = lamda
    config['model_ema'] = lamda_ema
    config['optimizer_primary'] = torch.optim.Adam(
            lamda.get_primary_params(),
            betas=(0.5, 0.999), lr=config['learning_rate'])
    config['optimizer_domain_disc'] = torch.optim.Adam(
            lamda.get_domain_disc_params(),
            betas=(0.5, 0.999), lr=config['learning_rate'])

    # make save_dir
    name = str(time()).split('.')[0]
    dir_name = config['save_dir'] + '/' + name
    os.makedirs(dir_name, exist_ok=True)
    config['save_dir'] = dir_name

    train(config, print_every=config['summary_freq'])
