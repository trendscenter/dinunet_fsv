#!/usr/bin/env python3

import json
import os
from os import sep

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams["figure.figsize"] = [16, 9]


def create_k_fold_splits(files, k=0, save_to_dir=None, shuffle_files=True):
    from random import shuffle
    import numpy as np

    if shuffle_files:
        shuffle(files)

    ix_splits = np.array_split(np.arange(len(files)), k)
    for i in range(len(ix_splits)):
        test_ix = ix_splits[i].tolist()
        val_ix = ix_splits[(i + 1) % len(ix_splits)].tolist()
        train_ix = [ix for ix in np.arange(len(files)) if ix not in test_ix + val_ix]

        splits = {'train': [files[ix] for ix in train_ix],
                  'validation': [files[ix] for ix in val_ix],
                  'test': [files[ix] for ix in test_ix]}

        if save_to_dir:
            f = open(save_to_dir + os.sep + 'SPLIT_' + str(i) + '.json', "w")
            f.write(json.dumps(splits))
            f.close()
        else:
            return splits


def init_k_folds(cache, state):
    """
    If one want to use custom splits:- Populate splits_dir as specified in inputs spec with split files(.json)
        with list of file names on each train, validation, and test keys.
    Number of split files should be equal to num_of_folds passed in inputspec
    If nothing is provided, random k-splits will be created.

    """
    out = {}
    split_dir = state['baseDirectory'] + sep + cache['split_dir']
    os.makedirs(split_dir, exist_ok=True)
    if len(os.listdir(split_dir)) == 0:
        create_k_fold_splits(files=os.listdir(state['baseDirectory'] + sep + cache['data_dir']),
                             k=cache['num_of_folds'],
                             save_to_dir=split_dir)
    elif len(os.listdir(split_dir)) != cache['num_of_folds']:
        raise ValueError(f"Number of splits in {split_dir} of site {state['clientId']} \
                         must be {cache['num_of_folds']} instead of {len(os.listdir(split_dir))}")

    splits = sorted(os.listdir(split_dir))
    cache['splits'] = dict(zip(range(len(splits)), splits))
    out['splits'] = {}
    for i, sp in cache['splits'].items():
        sp = json.loads(open(f"{split_dir}/{sp}").read())
        out['splits'][i] = len(sp['train'])
    out['batch_size'] = cache['batch_size']
    out['id'] = cache['id']
    return out


def safe_collate(batch):
    return default_collate([b for b in batch if b])


class NNDataLoader(DataLoader):

    def __init__(self, **kw):
        super(NNDataLoader, self).__init__(**kw)

    @classmethod
    def new(cls, **kw):
        _kw = {
            'dataset': None,
            'batch_size': 1,
            'shuffle': False,
            'sampler': None,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': None
        }
        for k in _kw.keys():
            _kw[k] = kw.get(k, _kw.get(k))
        return cls(collate_fn=safe_collate, **_kw)


def save_logs(cache, plot_keys=[], file_keys=[], num_points=51, log_dir=None):
    scaler = MinMaxScaler()
    for k in plot_keys:
        data = cache.get(k, [])

        if len(data) == 0:
            continue

        df = pd.DataFrame(data[1:], columns=data[0].split(','))

        if len(df) == 0:
            continue

        for col in df.columns:
            if max(df[col]) > 1:
                df[col] = scaler.fit_transform(df[[col]])

        rollin_window = max(df.shape[0] // num_points + 1, 3)
        rolling = df.rolling(rollin_window, min_periods=1).mean()
        ax = df.plot(x_compat=True, alpha=0.1, legend=0)
        rolling.plot(ax=ax, title=k.upper())

        plt.savefig(log_dir + os.sep + k + '.png')
        plt.close('all')

    for fk in file_keys:
        with open(log_dir + os.sep + f'{fk}.csv', 'w') as file:
            for line in cache[fk] if any(isinstance(ln, list)
                                         for ln in cache[fk]) else [cache[fk]]:
                if isinstance(line, list):
                    file.write(','.join([str(s) for s in line]) + '\n')
                else:
                    file.write(f'{line}\n')