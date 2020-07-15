import json
from os import sep

import torch
import torch.nn as nn

from classification import FreeSurferDataset
from core.models import MSANNet


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def save_checkpoint(cache, model, optimizer, id):
    chk = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(chk, cache['log_dir'] + sep + id)


def load_checkpoint(cache, model, optimizer, id):
    checkpoint = torch.load(cache['log_dir'] + sep + id)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def init_nn(cache, init_weights=False):
    """
    Initialize neural network/optimizer with locked parameters(check on remote script for that).
    Also detect and assign specified GPUs.
    @note Works only with one GPU per site at the moment.
    """
    if torch.cuda.is_available() and cache.get('use_gpu'):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = MSANNet(in_size=cache['input_size'], hidden_sizes=cache['hidden_sizes'],
                    out_size=cache['num_class'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cache['learning_rate'])
    if init_weights:
        torch.manual_seed(cache['seed'])
        initialize_weights(model)
    return {'device': device, 'model': model.to(device), 'optimizer': optimizer}


def init_dataset(cache, state):
    """
    Parse and load dataset and save to cache:
    so that in next global iteration we dont have to do that again.
    The data IO depends on use case-For a instance, if your data can fit in RAM, you can load
     and save the entire dataset in cache. But in general,
     it is better to save indices in cache and load only the mini-batch at a time
     (logic in __nextitem__) of the data loader.
    """
    dataset = FreeSurferDataset(files_dir=state['baseDirectory'] + sep + cache['data_dir'],
                                labels_file=state['baseDirectory'] + sep + cache['label_dir'],
                                mode=cache['mode'])
    split = json.loads(
        open(state['baseDirectory'] + sep + cache['split_dir'] + sep + cache['split_file']).read())
    dataset.load_indices(files=split['train'])
    cache['data_indices'] = dataset.indices
    if len(dataset) % cache['batch_size'] >= 4:
        cache['data_len'] = len(dataset)
    else:
        cache['data_len'] = (len(dataset) // cache['batch_size']) * cache['batch_size']


