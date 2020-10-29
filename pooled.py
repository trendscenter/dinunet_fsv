import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from coinstac_pyprofiler import custom_profiler as cprof
from torch.utils.data import ConcatDataset

from core.datasets import PooledDataset
from core.measurements import Prf1a
from core.utils import initialize_weights, NNDataLoader
from models import MSANNet

pooled_log_output_dir = "./pooled_log/pooled_epoch_101_bs_32"


def get_dataset(site, conf, fold, split_key=None):
    dataset = PooledDataset(data_dir=f"test/input/local{site}/simulatorRun/{conf['data_dir']['value']}",
                            label_dir=f"test/input/local{site}/simulatorRun/{conf['label_dir']['value']}",
                            mode=split_key)
    split_file = f"test/input/local{site}/simulatorRun/{conf['split_dir']['value']}/SPLIT_{fold}.json"
    split = json.loads(open(split_file).read())
    dataset.load_indices(split[split_key])
    return dataset


def eval(data_loader, model, device):
    model.eval()
    score = Prf1a()
    for i, batch in enumerate(data_loader):
        inputs, labels = batch['inputs'].to(device).float(), batch['labels'].to(device).long()
        if inputs.shape[0] > 1:
            out = F.log_softmax(model(inputs), 1)
            _, preds = torch.max(out, 1)
            sc = Prf1a()
            sc.add(preds, labels)
            score.accumulate(sc)
    return score


def train(fold, model, optim, device, args, train_loader, val_loader):
    best_score = 0.0
    best_ep = 0
    for ep in range(args['epochs']['value']):
        for i, batch in enumerate(train_loader):
            inputs, labels = batch['inputs'].to(device).float(), batch['labels'].to(device).long()

            optim.zero_grad()
            out = F.log_softmax(model(inputs), 1)
            wt = torch.randint(1, 101, (2,)).float()
            loss = F.nll_loss(out, labels, weight=wt)
            loss.backward()
            optim.step()

            _, preds = torch.max(out, 1)
            score = Prf1a()
            score.add(preds, labels)
            if i % int(math.log(i + 1) + 1) == 0:
                print(
                    f"Ep:{ep}/{args['epochs']['value']}, Itr:{i}/{len(train_loader)}, {round(loss.item(), 4)}, {score.prfa()}")
        val_score = eval(val_loader, model, device)
        if val_score.f1 > best_score:
            best_score = val_score.f1
            best_ep = ep
            torch.save(model.state_dict(), os.path.join(pooled_log_output_dir, f'best_{fold}.pt'))

            print(f'##### *** BEST saved ***  {best_score}')
        else:
            print('###### Not Improved:', val_score.f1, best_score)

        if ep - best_ep >= args['patience']['value']:
            break


@cprof.profile(type="pyinstrument", output_file_prefix=pooled_log_output_dir, params_dict={'save_html': True})
def start_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(pooled_log_output_dir, exist_ok=True)
    global_score = Prf1a()

    for fold in range(10):

        train_set, val_set = [], []
        for s, conf in enumerate(inputspecs):
            train_set.append(get_dataset(s, conf, fold, 'train'))
            val_set.append(get_dataset(s, conf, fold, 'validation'))

        model = nn.DataParallel(MSANNet(in_size=66, hidden_sizes=args['hidden_sizes']['value'], out_size=2))
        model = model.to(device)
        torch.manual_seed(args['seed']['value'] if args.get('seed') else 122331)
        initialize_weights(model)
        optim = torch.optim.Adam(model.parameters(), lr=args['learning_rate']['value'])

        run_test = args['mode']['value'] == 'test'
        if args['mode']['value'] == 'train':
            train_dset = ConcatDataset(train_set)
            train_loader = NNDataLoader.new(dataset=train_dset, batch_size=args['batch_size']['value'],
                                            pin_memory=True, shuffle=True, drop_last=True)
            val_dset = ConcatDataset(val_set)
            val_loader = NNDataLoader.new(dataset=val_dset, batch_size=args['batch_size']['value'], pin_memory=True,
                                          shuffle=True)
            print(f'Fold {fold}:', "Length of training dataset: ", len(train_dset), "Length of validation dataset: ",
                  +                  len(val_dset))
            train(fold, model, optim, device, args, train_loader, val_loader)
            run_test = True

        if run_test:
            test_set = []
            for s, conf in enumerate(inputspecs):
                test_set.append(get_dataset(s, conf, fold, 'test'))
            test_dset = ConcatDataset(test_set)
            test_loader = NNDataLoader.new(dataset=test_dset, batch_size=args['batch_size']['value'], pin_memory=True,
                                           shuffle=True)
            model.load_state_dict(torch.load(os.path.join(pooled_log_output_dir, f'best_{fold}.pt')))

            print(f'Fold {fold}:', "Length of validation dataset: ", len(test_dset))
            test_score = eval(test_loader, model, device)
            global_score.accumulate(test_score)
            with open(os.path.join(pooled_log_output_dir, f'{fold}_prfa.txt'), 'w') as wr:
                wr.write(f'{test_score.prfa()}')
                wr.flush()

    with open(os.path.join(pooled_log_output_dir, f'global_prfa.txt'), 'w') as wr:
        wr.write(f'{global_score.prfa()}')
        wr.flush()


if __name__ == "__main__":
    inputspecs = json.loads(open('test/inputspec.json').read())
    args = inputspecs[0]
    args['epochs']['value'] = 101
    args['batch_size']['value'] = 32
    args['patience']['value'] = 31
    # args['hidden_sizes']['value'] = [256, 128, 64, 32]

    start_training(args)
