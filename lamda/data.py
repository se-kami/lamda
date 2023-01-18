#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from scipy.io import loadmat


def load_mat(file):
    m = loadmat(file)

    x = torch.tensor(m['X'])
    y = torch.tensor(m['y']).view(-1).to(torch.long)
    y[y == 10] = 0

    # reshape to batch channel height width
    shape = x.shape
    h = shape.index(32)
    w = len(shape) - shape[::-1].index(32) - 1
    c = shape.index(3)
    b = sum(range(len(shape))) - h - w - c

    x = x.permute(b, c, h, w)
    if x.dtype == torch.uint8:
        x = x / 255
    return x, y


def load_csv(filename):
    df = pd.read_csv(filename)
    features = torch.tensor(df.iloc[:, :-1].to_numpy(), dtype=torch.float)
    labels = torch.tensor(df.iloc[:, -1].to_numpy(), dtype=torch.long)
    return features, labels


class DatasetCSV(Dataset):
    def __init__(self, csv_file):
        """
        csv_file: csv file of embeddings, last column is class label
        """
        super().__init__()

        self.features, self.labels = load_csv(csv_file)

        self.size = self.labels.shape[0]

        self.targets = torch.unique(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.size

    def get_number_of_classes(self):
        return len(self.targets)


class DatasetMat(Dataset):
    def __init__(self, mat_file):
        """
        mat_file: mat file of embeddings, images are under key 'X', labels under key 'y'
        """
        super().__init__()

        self.features, self.labels = load_mat(mat_file)

        self.size = self.labels.shape[0]

        self.targets = torch.unique(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.size

    def get_number_of_classes(self):
        return len(self.targets)


class DatasetImageFolder(Dataset):
    extensions = ('jpg', 'png', 'jpeg', 'webp', 'tif', 'tiff')

    def __init__(self, data_dir, transform=None,
                 extensions=None, le=None,):
        """
        data_dir: base directory of data
        transform: transformation to apply
        extensions: what filetypes to get
        le: label encoder
        """
        super().__init__()

        if transform is None:
            self.transform = get_image_transform()

        if extensions is None:
            extensions = DatasetImageFolder.extensions

        data = get_all_ext_in_dir(data_dir, extensions)

        # get images
        self.images = [img for img, _ in data]

        # get labels
        data_labels = [label for _, label in data]

        # encode labels
        self.le.fit(data_labels)
        self.labels = torch.tensor(self.le.transform(data_labels))

        self.size = self.labels.shape[0]
        self.targets = torch.unique(self.labels)

    def __getitem__(self, index):
        """
        returns transformed image and label
        """
        index = index % self.size

        # transform images
        img_name = self.images[index]
        if img_name.ends_with(('tif', 'tiff')):
            img = imread(img_name)
            img = Image.fromarray(img).convert('RGB')
        else:
            img = Image.open(img_name).convert("RGB")
        img = self.src_transform(img)

        # get labels
        label = self.labels[index]

        return img, label

    def __len__(self):
        return self.size

    def get_number_of_classes(self):
        return len(self.le.classes_)

    def get_names_of_images(self):
        return self.images


def get_image_transform(size=[224, 224],
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    resize, to tensor, normalize
    """
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


def get_all_ext_in_dir(directory, extensions):
    def get_group(filename):
        return str(filename).split('/')[-2] if not test else None
    # convert to tuple
    extensions = tuple(extensions)

    # find all images
    p = Path(directory)
    all_files = [(str(file), get_group(file))
                 for file in p.rglob("*")
                 if str(file).lower().endswith(extensions)]

    return all_files


class DoubleLoader():
    def __init__(self, ds1, ds2, batch_size1=1, batch_size2=1, shuffle1=False, shuffle2=False):
        self.l1 = DataLoader(ds1, batch_size=batch_size1, shuffle=shuffle1)
        self.l2 = DataLoader(ds2, batch_size=batch_size2, shuffle=shuffle2)
        self.size1 = len(ds1)
        self.size2 = len(ds2)
        self.size = max(self.size1, self.size2)
        self.n_classes = max(ds1.get_number_of_classes(),
                             ds2.get_number_of_classes(),)

        self.iter_1 = self._get_iter_1()
        self.iter_2 = self._get_iter_2()
        self.i = 0
        self.step = max(batch_size1, batch_size2)
        self.in_size = tuple(ds1[0][0].shape)

    def __iter__(self):
        self.iter_1 = self._get_iter_1()
        self.iter_2 = self._get_iter_2()
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        else:
            self.i += self.step
        try:
            data_1 = next(self.iter_1)
        except StopIteration:
            self.iter_1 = self._get_iter_1()  # reset iter
            data_1 = next(self.iter_1)

        try:
            data_2 = next(self.iter_2)
        except StopIteration:
            self.iter_2 = self._get_iter_2()  # reset iter
            data_2 = next(self.iter_2)

        # make batches be the same size
        if len(data_1) != len(data_2):
            size = min(len(data_1), len(data_2))
            data_1 = data_1[:size]
            data_2 = data_2[:size]

        return data_1, data_2

    def _get_iter_1(self):
        return iter(self.l1)

    def _get_iter_2(self):
        return iter(self.l2)

    def get_number_of_classes(self):
        return self.n_classes

    def get_in_size(self):
        return self.in_size

    def __len__(self):
        return self.size


class RandomSampler(DoubleLoader):
    def __init__(self, ds1, ds2, batch_size=1):
        self.ds1 = ds1
        self.ds2 = ds2
        self.batch_size = batch_size
        self.size_1 = len(self.ds1)
        self.size_2 = len(self.ds2)
        self.n_classes = max(ds1.get_number_of_classes(),
                             ds2.get_number_of_classes(),)
        self.in_size = tuple(ds1[0][0].shape)


    def __call__(self):
        ds1 = self.ds1[torch.randperm(self.size_1)[:self.batch_size]]
        ds2 = self.ds2[torch.randperm(self.size_2)[:self.batch_size]]
        return ds1, ds2


def get_datasets(source_train, source_test, target_train, target_test,
                 data_dir=""):
    """
    return src_train, src_test, trg_train, trg_test
    """
    # remove trailing /
    if len(data_dir) == 0:
        data_dir = '.'
    if data_dir[-1] == '/':
        data_dir = data_dir[:-1]

    sources = [('source_train', source_train),
               ('source_test', source_test),
               ('target_train', target_train),
               ('target_test', target_test)]
    sources = [source_train, source_test, target_train, target_test]

    sources = [f"{data_dir}/{source}" for source in sources]
    datasets = []
    for source in sources:
        if source.endswith('.csv'):
            ds = DatasetCSV
        elif source.endswith('.mat'):
            ds = DatasetMat
        else:
            ds = DatasetImageFolder
        datasets.append(ds)
    datasets = [d(s) for d, s in zip(datasets, sources)]

    return datasets


def get_dataset(dataset, data_dir=""):
    # remove trailing /
    if len(data_dir) == 0:
        data_dir = '.'
    if data_dir[-1] == '/':
        data_dir = data_dir[:-1]

    dataset = f"{data_dir}/{dataset}"
    if dataset.endswith('.csv'):
        ds = DatasetCSV
    elif dataset.endswith('.mat'):
        ds = DatasetMat
    else:
        ds = DatasetImageFolder
    dataset = ds(dataset)

    return dataset


def get_loaders(source_train, source_test, target_train, target_test,
                batch_size_train=128, batch_size_test=128,
                shuffle_train=True, shuffle_test=False,
                infinite_sampler=True, data_dir=""):
    """
    infinite_sampler
        if False: dataloader_train is normal pytorch dataloader
        if True: dataloader_train is infinite random sampler
    """
    datasets = get_datasets(source_train=source_train,
                            source_test=source_test,
                            target_train=target_train,
                            target_test=target_test,
                            data_dir=data_dir)

    single_loaders = [
            DataLoader(datasets[0],
                       batch_size=batch_size_train, shuffle=shuffle_train),
            DataLoader(datasets[2],
                       batch_size=batch_size_train, shuffle=shuffle_train),
            DataLoader(datasets[1],
                       batch_size=batch_size_test, shuffle=shuffle_test),
            DataLoader(datasets[3],
                       batch_size=batch_size_test, shuffle=shuffle_test),
            ]

    if infinite_sampler:
        dataloader_train = RandomSampler(datasets[0], datasets[2], batch_size_train)
    else:
        dataloader_train = DoubleLoader(datasets[0], datasets[2],
                                        batch_size_train, batch_size_train,
                                        shuffle_train, shuffle_train)

    return dataloader_train, single_loaders


if __name__ == '__main__':
    l1, l2 = get_loaders('amazon_train.csv', 'amazon_test.csv', 'webcam_train.csv', 'webcam_test.csv', 128, 128, False, False, 'data')
    for x, y in l1:
        print(len(x[0]))
        break
