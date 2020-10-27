import os

import pandas as pd
import torch

from core.utils import NNDataset


class FreeSurferDataset(NNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)

    def load_indices(self, files, **kw):
        labels_file = os.listdir(self.label_dir)[0]
        labels = pd.read_csv(self.label_dir + os.sep + labels_file).set_index('freesurferfile')
        for file in files:
            y = labels.loc[file]['label']
            """
            int64 could not be json serializable.
            """
            self.indices.append([file, int(y)])

    def __getitem__(self, ix):
        file, y = self.indices[ix]
        df = pd.read_csv(self.data_dir + os.sep + file, sep='\t', names=['File', file], skiprows=1)
        df = df.set_index(df.columns[0])
        x = df.T.iloc[0].values
        return {'inputs': torch.tensor(x), 'labels': torch.tensor(y), 'ix': torch.tensor(ix)}


class PooledDataset(FreeSurferDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.data_dir = kw['data_dir']
        self.label_dir = kw['label_dir']
        self.mode = kw['mode']
