import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from coinstac_dinunet import COINNDataset, COINNTrainer, COINNDataHandle
from coinstac_dinunet.metrics import AUCROCMetrics

from .models import ICALstm


# input_size=256,
# hidden_size=256,
# num_layers=1,
# num_cls=2,
# bidirectional=True,
# proj_size=64, num_comps=53,
# window_size=20,
# seq_len=7

def read_lines(file):
    return np.array([int(float(l.strip())) for l in open(file).readlines()])


class ICADataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = None
        self.data = None
        self.full_comp_size = self.cache.setdefault('full_comp_size', 100)
        self.spatial_dim = self.cache.setdefault('spatial_dim', 140)
        self.window_size = self.cache.setdefault('window_size', 20)
        self.window_stride = self.cache.setdefault('window_stride', 10)
        self.seq_len = self.cache.setdefault('seq_len', 13)
        self.h5py_key = self.cache['data_file'].split('_')[0] + "_dataset"

    def load_index(self, ix):
        if self.data is None:
            self.labels = pd.read_csv(self.state['baseDirectory'] + os.sep + self.cache['labels_file'])
            self.labels = self.labels.set_index('index')

            hf = h5py.File(self.path(cache_key='data_file'), "r")
            data = np.array(hf.get(self.h5py_key))
            data = data.reshape((data.shape[0], self.full_comp_size, self.spatial_dim))

            if self.cache.get('components_file'):
                use_ix = read_lines(self.path(cache_key='components_file'))
                data = data[:, use_ix, :]

            data = torch.from_numpy(data)
            n_comp = data.shape[1]
            unfold = nn.Unfold(kernel_size=(1, self.window_size), stride=(1, self.window_stride))
            data = unfold(data.unsqueeze(2)).reshape(data.shape[0], -1, n_comp, self.window_size)

            assert (data.shape[1] == self.seq_len), \
                f"Sequence len did not match: {data.shape[1]} vs {self.cache['seq_len']}"
            self.data = data

        y = self.labels.loc[ix][0]

        """int64 could not be json serializable.  """
        self.indices.append([ix, int(y - 1)])

    def __getitem__(self, ix):
        data_index, y = self.indices[ix]
        return {'inputs': self.data[data_index].clone(), 'labels': y}


class ICATrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['net'] = ICALstm(
            num_layers=self.cache.setdefault('num_layers', 1),
            input_size=self.cache.setdefault('input_size', 512),
            seq_len=self.cache.setdefault('seq_len', 13),
            hidden_size=self.cache.setdefault('hidden_size', 384),
            proj_size=self.cache.setdefault('proj_size', 128),
            num_cls=self.cache.setdefault('num_class', 2)
        )

    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()

        out, h = self.nn['net'](inputs)
        out = F.log_softmax(out, 1)
        loss = F.nll_loss(out, labels)

        _, pred = torch.max(out, 1)
        score = self.new_metrics()
        score.add(out[:, 1], labels)

        val = self.new_averages()
        val.add(loss.item(), len(inputs))

        return {'out': out, 'loss': loss, 'averages': val, 'metrics': score, 'prediction': pred}

    def new_metrics(self):
        return AUCROCMetrics()


class ICADataHandle(COINNDataHandle):
    def list_files(self):
        ix = list(
            pd.read_csv(self.state['baseDirectory'] + os.sep + self.cache['labels_file'])['index'].tolist()
        )
        return ix
