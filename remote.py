#!/usr/bin/python
import json
import os
import sys

import numpy as np
import pydevd_pycharm

pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)


def aggregate_sites_grad(input):
    out = {}
    grads = []
    for site, site_vars in input.items():
        grad_file = state['baseDirectory'] + os.sep + site + os.sep + site_vars['grads_file']
        grad = np.load(grad_file, allow_pickle=True)
        grads.append(grad)
    out['avg_grads_file'] = 'avg_grads.npy'
    avg_grads = []
    for layer_grad in zip(*grads):
        avg_grads.append(np.array(layer_grad).mean(0))
    np.save(state['transferDirectory'] + os.sep + out['avg_grads_file'], np.array(avg_grads))
    return out


def on_epoch_end():
    out = {}
    return out


def init_nn_params():
    out = {}
    out['input_size'] = 66
    out['hidden_sizes'] = [16, 8, 4, 2]
    out['num_class'] = 2
    out['epochs'] = 11
    out['learning_rate'] = 0.001
    return out


def generate_folds(cache, input):
    cache['folds'] = list(zip(*[site['splits'] for _, site in input.items()]))


def next_run(cache, input):
    out = {}
    out['fold'] = set(cache['folds'].pop()).pop()
    out['seed'] = 244627
    out['batch_size'] = 16  # Todo
    return out


def check(logic, k, v, input):
    phases = []
    for site_vars in input.values():
        phases.append(site_vars.get(k) == v)
    return logic(phases)


def set_mode(input, mode=None):
    out = {}
    for site, site_vars in input.items():
        out[site] = mode if mode else site_vars['mode']
    return out


def save_test_scores():
    pass


if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    cache = args['cache']
    input = args['input']
    state = args['state']
    out = {}

    nxt_phase = input.get('phase', 'init_runs')
    if check(all, 'phase', 'init_runs', input):
        cache.update(train_log=['Loss,Precision,Recall,F1,Accuracy'],
                     validation_log=['Loss,Precision,Recall,F1,Accuracy'])
        out['nn'] = init_nn_params()
        generate_folds(cache, input)
        out['run'] = next_run(cache, input)
        nxt_phase = 'init_nn'
        out['global_modes'] = set_mode(input)

    if check(all, 'phase', 'computation', input):
        nxt_phase = 'computation'
        if check(any, 'mode', 'train', input):
            out.update(**aggregate_sites_grad(input))
            out['global_modes'] = set_mode(input)

        if check(all, 'mode', 'val_waiting', input):
            out['global_modes'] = set_mode(input, mode='validation')

        if check(all, 'mode', 'train_waiting', input):
            out.update(**on_epoch_end())
            out['global_modes'] = set_mode(input, mode='train')

        if check(all, 'mode', 'test', input):
            out.update(**on_epoch_end())
            out['global_modes'] = set_mode(input, mode='test')

    if check(all, 'phase', 'next_run_waiting', input):
        if len(cache['folds']) > 0:
            out['run'] = next_run(cache, input)
            nxt_phase = 'init_nn'
            out['global_modes'] = set_mode(input)
        else:
            out['phase'] = 'success'

    out['phase'] = nxt_phase
    output = json.dumps({'output': out, 'cache': cache, 'success': check(all, 'phase', 'success', input)})
    sys.stdout.write(output)
