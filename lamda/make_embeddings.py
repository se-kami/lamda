#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from data import get_dataset
import pandas as pd


def make_embeddings(lamda, dataset, device, batch_size=310):
    loader = DataLoader(dataset, batch_size=batch_size)
    xs, ys = [], []
    for x, y in loader:
        xs.append(lamda.embed(x.to(device)).detach())
        ys.append(y)
    xs = torch.cat(xs).cpu().numpy()
    ys = torch.cat(ys).numpy()
    df = pd.DataFrame(xs)
    df['label'] = ys
    return df

def save_df(df, filename):
    df.to_csv(filename, header=False, index=False)

import torch
from model import LAMDA

lamda = LAMDA(model_size='small', in_size=(2048,), out_size=31, share_encoders=True)
device = torch.device('cuda')
lamda = lamda.to(device)
lamda.load_models('runs/1670454660/lamda-best001600.pth')

dataset = get_dataset('data/amazon_test.csv')
df = make_embeddings(lamda, dataset, device)
save_df(df, 'embeddings_amazon.csv')

dataset = get_dataset('data/dslr_test.csv')
df = make_embeddings(lamda, dataset, device)
save_df(df, 'embeddings_dslr.csv')
