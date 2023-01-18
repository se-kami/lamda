#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.manifold import TSNE
import torch
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
plt.style.use('fivethirtyeight')

def load_embeddings(filename):
    df = pd.read_csv(filename)
    features = torch.tensor(df.iloc[:, :-1].to_numpy(), dtype=torch.float)
    labels = torch.tensor(df.iloc[:, -1].to_numpy(), dtype=torch.long)
    return features, labels


def plot_tsne(emb_src, emb_trg, filename):
    src_x, src_y = load_embeddings(emb_src)
    trg_x, trg_y = load_embeddings(emb_trg)

    n_classes = len(set(src_y).union(set(trg_y)))

    color_map = get_colormap(n_classes)
    color_map = matplotlib.colormaps['hsv_r']
    colors_src = [color_map(255*i/n_classes) for i in src_y.numpy()]
    colors_trg = [color_map(255*i/n_classes) for i in trg_y.numpy()]

    colors_src = [i for i in src_y.numpy()]
    colors_trg = [i for i in trg_y.numpy()]

    xs, ys = zip(*TSNE(learning_rate=20, perplexity=40).fit_transform(torch.cat([src_x, trg_x])))

    fig, ax = plt.subplots(figsize=(12, 8), dpi=500)
    ax.scatter(xs[:len(src_x)], ys[:len(src_x)], c=colors_src, s=1, marker='o', label='source', cmap='rainbow_r')
    ax.scatter(xs[len(src_x):], ys[len(src_x):], c=colors_trg, s=.25, marker='x', label='target', cmap='rainbow_r')
    ax.set_title('T-SNE visualization')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(visible=False)
    ax.legend()
    fig.savefig(filename)


def get_colormap(n_classes):
    colors = dict()

if __name__ == '__main__':
    plot_tsne('data/amazon_train.csv', 'data/dslr_train.csv', 'tsne-resnet.png')
    plot_tsne('embeddings_amazon.csv', 'embeddings_dslr.csv', 'tsne-lamda.png')
