#!/usr/bin/python

import os
import pandas as pd
import torch
from coinstac_dinunet import COINNDataset, COINNTrainer, COINNLocal
from coinstac_dinunet.metrics import Prf1a
import torch.nn.functional as F
from coinstac_dinunet.io import RECV

from model import MSANNet


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)


class FreeSurferDataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = {}

    def load_index(self, site, file):
        if self.labels.get(site) is None:
            label_dir = self.path(site, 'label_dir')
            labels_file = os.listdir(label_dir)[0]
            self.labels[site] = pd.read_csv(label_dir + os.sep + labels_file).set_index('freesurferfile')
        y = self.labels[site].loc[file]['label']
        """
        int64 could not be json serializable.
        """
        self.indices.append([site, file, int(y)])

    def __getitem__(self, ix):
        site, file, y = self.indices[ix]
        data_dir = self.path(site, 'data_dir')
        df = pd.read_csv(data_dir + os.sep + file, sep='\t', names=['File', file], skiprows=1)
        df = df.set_index(df.columns[0])
        df = df / df.max().astype('float64')
        x = df.T.iloc[0].values
        return {'inputs': torch.tensor(x), 'labels': torch.tensor(y), 'ix': torch.tensor(ix)}


class FreeSurferTrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['fs_net'] = MSANNet(in_size=self.cache['input_size'],
                                    hidden_sizes=self.cache['hidden_sizes'], out_size=self.cache['num_class'])

    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()
        indices = batch['ix'].to(self.device['gpu']).long()

        out = F.log_softmax(self.nn['fs_net'](inputs), 1)
        wt = torch.randint(1, 101, (2,)).to(self.device['gpu']).float()
        loss = F.nll_loss(out, labels, weight=wt)

        _, predicted = torch.max(out, 1)
        score = self.new_metrics()
        score.add(predicted, labels)
        val = self.new_averages()
        val.add(loss.item(), len(inputs))
        return {'out': out, 'loss': loss, 'averages': val, 'metrics': score, 'prediction': predicted,
                'indices': indices}

    def _set_monitor_metric(self):
        self.cache['monitor_metric'] = 'f1', 'maximize'

    def _set_log_headers(self):
        self.cache['log_header'] = 'Loss|Accuracy,F1'

    def new_metrics(self):
        return Prf1a()


if __name__ == "__main__":
    pretrain_args = {'epochs': 51, 'batch_size': 16}
    local = COINNLocal(cache=RECV['cache'], input=RECV['input'], pretrain_args=pretrain_args, batch_size=16,
                       state=RECV['state'], epochs=111, patience=21, computation_id='fsv_quick')
    local.compute(FreeSurferDataset, FreeSurferTrainer)
    local.send()
