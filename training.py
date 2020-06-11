# !/usr/bin/python

import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from core import data_parser
from core.torchutils import NNDataLoader

sep = os.sep
from core.measurements import Prf1a


# #
# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)


class FreeSurferDataset(Dataset):
    def __init__(self, **kw):
        self.files_dir = kw['files_dir']
        self.labels_dir = kw['labels_file']
        self.mode = kw['mode']
        self.indices = []

    def load_indices(self, files, **kw):
        labels_file = os.listdir(self.labels_dir)[0]
        labels = pd.read_csv(self.labels_dir + os.sep + labels_file).set_index('freesurferfile')
        for file in files:
            y = labels.loc[file]['label']
            """
            int64 could not be json serializable.
            """
            self.indices.append([file, int(y)])

    def __getitem__(self, ix):
        file, y = self.indices[ix]
        data, errors = data_parser.parse_subj_volume_files(self.files_dir, [file])
        x = data.iloc[0].values
        return {'inputs': torch.tensor(x), 'labels': torch.tensor(y)}

    def __len__(self):
        return len(self.indices)

    def get_loader(self, shuffle=False, batch_size=None, num_workers=0, pin_memory=True, **kw):
        return NNDataLoader.get_loader(dataset=self, shuffle=shuffle, batch_size=batch_size,
                                       num_workers=num_workers, pin_memory=pin_memory, **kw)


def single_shot_training(cache, state, model, optimizer, **kw):
    dataset = FreeSurferDataset(files_dir=state['baseDirectory'] + sep + cache['data_dir'],
                                labels_file=state['baseDirectory'] + sep + cache['label_dir'],
                                mode=cache['mode'])
    split = json.loads(open(state['baseDirectory'] + sep + cache['split_dir'] + sep + cache['split_file']).read())
    dataset.load_indices(files=split['train'])
    dataloader = dataset.get_loader(shuffle=True, batch_size=cache['batch_size'], drop_last=True,
                                    num_workers=cache.get('num_workers', 0), pin_memory=cache.get('pin_memory', True))

    for ep in range(cache['epochs']):
        model.train()
        running_loss = 0
        score = Prf1a()
        for i, batch in enumerate(dataloader):
            inputs, labels = batch['inputs'].to(kw['device']).float(), batch['labels'].to(kw['device']).long()

            optimizer.zero_grad()
            out = F.log_softmax(model(inputs), 1)
            loss = F.nll_loss(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.shape[0]

            _, predicted = torch.max(out, 1)
            score.add_tensor(predicted, labels)

        cache['training_log'].append([running_loss / len(dataset), *score.prfa()])
        val_loss, val_score = evaluation(cache, state, model, split_key='validation', device=kw['device'])
        if val_score.f1 > cache.get(f'best_score', 0):
            cache['best_score'] = val_score.f1
            state_dict = {'model_state_dict': model.state_dict()}
            torch.save(state_dict, cache['log_dir'] + os.sep + cache['best_nn_state'])
        cache['validation_log'].append([val_loss, *val_score.prfa()])


def get_next_batch(cache, state):
    dataset = FreeSurferDataset(files_dir=state['baseDirectory'] + sep + cache['data_dir'],
                                labels_file=state['baseDirectory'] + sep + cache['label_dir'],
                                mode=cache['mode'])
    if cache['resume_from_state']:
        dataset.indices = cache['data_indices'][cache['data_cursor']:]
    else:
        split = json.loads(open(state['baseDirectory'] + sep + cache['split_dir'] + sep + cache['split_file']).read())
        dataset.load_indices(files=split['train'])
        cache['data_indices'] = dataset.indices[0:len(dataset) // cache['batch_size'] * cache['batch_size']] \
            if len(dataset) % cache['batch_size'] < 4 else dataset.indices

    loader = dataset.get_loader(batch_size=cache['batch_size'],
                                num_workers=cache.get('num_workers', 0),
                                pin_memory=cache.get('pin_memory', True))
    return next(loader.__iter__())


def global_nn_iteration(cache, state, model, optimizer, **kw):
    model.train()

    batch = get_next_batch(cache, state)

    inputs, labels = batch['inputs'].to(kw['device']).float(), batch['labels'].to(kw['device']).long()
    optimizer.zero_grad()
    out = F.log_softmax(model(inputs), 1)
    loss = F.nll_loss(out, labels)
    loss.backward()
    if kw.get('avg_grads') is not None:
        for i, param in enumerate(model.parameters()):
            tensor = torch.FloatTensor(kw.get('avg_grads')[i]).to(kw['device'])
            param.grad.data = torch.autograd.Variable(tensor)
        optimizer.step()
    _, predicted = torch.max(out, 1)

    score = Prf1a()
    score.add_tensor(predicted, batch['labels'].long())
    cache['running_log'].append([loss.item() * inputs.shape[0], score.tp, score.fp, score.tn, score.fn])

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, cache['log_dir'] + sep + cache['current_nn_state'])


def evaluation(cache, state, model, split_key, **kw):
    model.eval()
    running_loss = 0
    eval_score = Prf1a()
    with torch.no_grad():
        val_dataset = FreeSurferDataset(files_dir=state['baseDirectory'] + sep + cache['data_dir'],
                                        labels_file=state['baseDirectory'] + sep + cache['label_dir'],
                                        mode=cache['mode'])
        split = json.loads(open(state['baseDirectory'] + sep + cache['split_dir'] + sep + cache['split_file']).read())
        val_dataset.load_indices(files=split[split_key])
        val_dataloader = val_dataset.get_loader(shuffle=False, batch_size=cache['batch_size'],
                                                num_workers=cache.get('num_workers', 0),
                                                pin_memory=cache.get('pin_memory', True))
        for batch in val_dataloader:
            inputs, labels = batch['inputs'].to(kw['device']).float(), batch['labels'].to(kw['device']).long()
            out = F.log_softmax(model(inputs), 1)
            _, predicted = torch.max(out, 1)
            loss = F.nll_loss(out, labels)
            running_loss += loss.item() * inputs.shape[0]
            eval_score.add_tensor(predicted, labels)
    return running_loss / len(val_dataset), eval_score
