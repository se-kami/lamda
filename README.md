# LAMDA: Label Matching Deep Domain Adaptation
This repository contains (un)official PyTorch implementation of the LAMDA paper.

## Architecture
![LAMDA Architecture](https://github.com/se-kami/lamda/blob/master/images/architecture.png | width=100)

## Embeddings
![T-SNE visualization of embeddings](https://github.com/se-kami/lamda/blob/master/images/embeddings.png | width=100)

## Setup
### Install dependencies.
```shell
pip install -r requirements.txt
```
### Download data
Download data into `<directory>` and rename appropriate fields in config files.
Default data directory is `./data/`

## Training
Configuration files for training Office-31 adaptation tasks are in `configs/configs-office31`.
Configuration files for training Digits adaptation tasks are in `configs/configs-digits`.
To train run
```shell
python lamda/train.py config_file.json
```

Logs are saved to `./runs/<run>` by default.

## Results
| Methods               | **A** --> **W** | **A** --> **D** | **D** --> **W** | **W** --> **D** | **D** --> **A** | **W** --> **A** |   Avg    |
| :-----------:         | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :------: |
| ResNet-50 [1]         |      70.0       |      65.5       |      96.1       |      99.3       |      62.8       |      60.5       |   75.7   |
| DeepCORAL [2]         |      83.0       |      71.5       |      97.9       |      98.0       |      63.7       |      64.5       |   79.8   |
| DANN [3]              |      81.5       |      74.3       |      97.1       |      99.6       |      65.5       |      63.2       |   80.2   |
| ADDA [4]              |      86.2       |      78.8       |      96.8       |      99.1       |      69.5       |      68.5       |   83.2   |
| CDAN [5]              |      94.1       |      92.9       |      98.6       |    **100.0**    |      71.0       |      69.3       |   87.7   |
| TPN [6]               |      91.2       |      89.9       |      97.7       |      99.5       |      70.5       |      73.5       |   87.1   |
| DeepJDOT [7]          |      88.9       |      88.2       |      98.5       |      99.6       |      72.1       |      70.1       |   86.2   |
| RWOT [8]              |      95.1       |      94.5       |      99.5       |    **100.0**    |      77.5       |      77.9       |   90.8   |
| **LAMDA Offical**     |    **95.2**     |    **96.0**     |      98.5       |    **100.0**    |    **87.3**     |    **84.4**     | **93.0** |
| **LAMDA (this repo)** |    **95.9**     |    **96.0**     |      98.6       |    **100.0**    |    **87.3**     |    **84.3**     | **93.2** |

## Citation

```
@InProceedings{pmlr-v139-le21a,
  title = 	 {LAMDA: Label Matching Deep Domain Adaptation},
  author =       {Le, Trung and Nguyen, Tuan and Ho, Nhat and Bui, Hung and Phung, Dinh},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {6043--6054},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/le21a/le21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/le21a.html},
  abstract = 	 {Deep domain adaptation (DDA) approaches have recently been shown to perform better than their shallow rivals with better modeling capacity on complex domains (e.g., image, structural data, and sequential data). The underlying idea is to learn domain invariant representations on a latent space that can bridge the gap between source and target domains. Several theoretical studies have established insightful understanding and the benefit of learning domain invariant features; however, they are usually limited to the case where there is no label shift, hence hindering its applicability. In this paper, we propose and study a new challenging setting that allows us to use a Wasserstein distance (WS) to not only quantify the data shift but also to define the label shift directly. We further develop a theory to demonstrate that minimizing the WS of the data shift leads to closing the gap between the source and target data distributions on the latent space (e.g., an intermediate layer of a deep net), while still being able to quantify the label shift with respect to this latent space. Interestingly, our theory can consequently explain certain drawbacks of learning domain invariant features on the latent space. Finally, grounded on the results and guidance of our developed theory, we propose the Label Matching Deep Domain Adaptation (LAMDA) approach that outperforms baselines on real-world datasets for DA problems.}
}
```

## References:
- https://github.com/tuanrpt/LAMDA
- https://proceedings.mlr.press/v139/le21a.html
