#!/usr/bin/python


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)

import json
import os

import numpy as np
import torch

from core import utils
from core.models import MSANNet

sep = os.sep
import sys
from core.measurements import Prf1a
import training as cc
import random
import shutil


def send_to_remote(output, cache, state, model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.numpy())

    output['grad_file'] = f"grad_{cache['current_nn_state']}.npy"
    np.save(state['transferDirectory'] + sep + output['grad_file'], np.array(grads))


def receive_from_remote(input, state):
    if input.get('avg_grad_file'):
        return np.load(state['baseDirectory'] + sep + input['avg_grad_file'], allow_pickle=True)


def init(cache, state):
    if input.get('global_nn_state'):
        cache.update(mode=input['global_nn_state'][state['clientId']]['mode'])

    cache.set(shuffle=False, data_cursor=0)
    cache.set(epoch=0, iteration=0)

    cache.set(training_log=['Loss,Precision,Recall,F1,Accuracy'],
              validation_log=['Loss,Precision,Recall,F1,Accuracy'],
              test_log=['Precision,Recall,F1,Accuracy'])

    cache.set(running_log=[])
    cache.set(validation_scores=[], test_scores=[])

    log_dir = state['outputDirectory'] + sep + cache['experiment'] + str(cache['which_k_fold'])
    os.makedirs(log_dir, exist_ok=True)

    cache.set(log_dir=log_dir, split_file="SPLIT_" + str(cache['which_k_fold']) + ".json")
    cache.set(log_key=cache.get('log_key', cache['split_file'].split('.')[0]))

    cache.set(current_nn_state=f"{cache['log_key']}_{cache['iteration']}_{cache['epoch']}.CHK",
              best_nn_state=cache['log_key'] + '.BEST')


def multi_shot_training(cache, input, output, state):
    if cache['mode'] in ['train', 'validation_waiting']:
        data_exhausted = cache.get('data_indices', False) \
                         and cache['data_cursor'] >= len(cache['data_indices'])
        if data_exhausted:
            cache.update(mode='validation_waiting', data_cursor=0)
            random.shuffle(cache['data_indices'])

        avg_grads = receive_from_remote(input, state)
        cc.global_nn_iteration(cache, state, model, optimizer, avg_grads=avg_grads, device=device)
        send_to_remote(output, cache, state, model)
        cache.update(iteration=cache['iteration'] + 1,
                     data_cursor=cache['data_cursor'] + cache['batch_size'])

    elif cache['mode'] == 'validation':
        score = Prf1a()
        (loss, tp, fp, tn, fn) = np.array(cache['running_log']).sum(0)
        score.add(tn=tn, fn=fn, tp=tp, fp=fp)
        cache['training_log'].append([loss / len(cache['data_indices'])] + list(score.prfa()))

        val_loss, val_score = cc.evaluation(cache, state, model, split_key='validation', device=device)
        cache['validation_log'].append([val_loss] + list(val_score.prfa()))
        output['validation_scores'] = [val_score.tp, val_score.tn, val_score.fp, val_score.fn]

        cache['mode'] = 'test' if cache['epoch'] >= cache['epochs'] else 'train_waiting'
        if cache['mode'] == 'train_waiting':
            cache.update(iteration=0, data_cursor=0, epoch=cache['epoch'] + 1, running_log=[])
            random.shuffle(cache['data_indices'])


def resume_nn_state(cache, model, optimizer):
    checkpoint_exists = os.path.exists(cache['log_dir'] + sep + cache['current_nn_state'])
    cache['resume_from_state'] = (cache['epoch'] > 0 and checkpoint_exists) or (
            cache['iteration'] > 0 and checkpoint_exists)
    if cache['resume_from_state']:
        checkpoint = torch.load(cache['log_dir'] + sep + cache['current_nn_state'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        torch.manual_seed(cache.get('torch_seed'))
        utils.initialize_weights(model)


def test(output, cache, state, model):
    model.load_state_dict(torch.load(cache['log_dir'] + sep + cache['best_nn_state'])['model_state_dict'])
    test_loss, _scores = cc.evaluation(cache, state, model, split_key='test', device=device)
    output['test_prf1a'] = _scores.prfa()
    output["test_scores"] = [_scores.tp, _scores.tn, _scores.fp, _scores.fn]
    cache['test_log'].append(_scores.prfa())
    utils.save_logs(cache, plot_keys=['training_log', 'validation_log'], file_keys=['test_log'])
    output['success'] = True


if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    cache = utils.Cache(args, with_input=True)
    input = args['input']
    state = args['state']

    """
    #### Hard coded defaults ######
    """
    cache['which_k_fold'] = 0
    cache.update(epochs=11, learning_rate=0.001, batch_size=16)
    cache["torch_seed"] = 244627
    """
    ###############################
    """
    init(cache, state)
    output = {}

    device = torch.device(cache.get('device', 'cpu'))
    model = MSANNet(in_size=cache['input_size'], hidden_sizes=cache['hidden_sizes'], out_size=cache['num_class'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cache['learning_rate'])

    if input.get('save_best_model'):
        shutil.copy(cache['log_dir'] + os.sep + cache['current_nn_state'],
                    cache['log_dir'] + os.sep + cache['best_nn_state'])

    if cache['mode'] == 'test':
        test(output, cache, state, model)

    if cache['experiment'] == 'single_shot':
        cc.single_shot_training(cache, state, model, optimizer, device=device)
        cache['mode'] = 'test'

    elif cache['experiment'] == 'multi_shot':
        resume_nn_state(cache, model, optimizer)
        multi_shot_training(cache, input, output, state)

    output['iteration'] = cache['iteration']
    output['epoch'] = cache['epoch']
    output['epochs'] = cache['epochs']
    output['mode'] = cache['mode']
    output['experiment'] = cache['experiment']
    output['which_k_fold'] = cache['which_k_fold']

    output = json.dumps({'output': output, 'cache': cache})
    sys.stdout.write(output)
