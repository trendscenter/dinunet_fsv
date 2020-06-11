# !/usr/bin/python
#
# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)

import json
import os
import sys

import numpy as np

from core.measurements import Prf1a
from core.utils import Cache, save_logs


def init(cache, inputs, state):
    cache.set(best_val_score=0, validation_log=['Precision,Recall,F1,Accuracy'],
              test_log=['Precision,Recall,F1,Accuracy'])
    log_dir = state['outputDirectory'] + os.sep + '_'.join(
        set([site_vars['experiment'] + str(site_vars.get('which_k_fold', '')) for _, site_vars in inputs.items()]))
    os.makedirs(log_dir, exist_ok=True)
    cache.set(log_dir=log_dir)
    cache.set(num_sites=len(inputs))
    cache.set(init=True)


def aggregate_scores(inputs, key):
    scores = []
    for site, site_vars in inputs.items():
        scores.append(site_vars[key])
    tp, tn, fp, fn = np.array(scores).sum(0).tolist()
    score = Prf1a()
    score.add(tp=tp, tn=tn, fn=fn, fp=fp)
    return score


def aggregate_sites_grad(input, sites_nn_state):
    grads = []
    for site, site_vars in input.items():
        if site_vars['mode'] == 'train':
            grad_file = state['baseDirectory'] + os.sep + site + os.sep + site_vars['grad_file']
            grad = np.load(grad_file, allow_pickle=True)
            grads.append(grad)
            sites_nn_state['mode'] = 'train'

    avg_grads = []
    for layer_grad in zip(*grads):
        avg_grads.append(np.array(layer_grad).mean(0))
    return avg_grads


if __name__ == "__main__":
    args = json.loads(sys.stdin.read())

    input = args['input']
    state = args['state']
    cache = Cache(args, with_input=False)

    init(cache, input, state)

    sites_nn_state = {}
    for site, site_vars in input.items():
        sites_nn_state[site] = {'iteration': site_vars['iteration'], 'epoch': site_vars['epoch'],
                                'epochs': site_vars['epochs'], 'mode': site_vars['mode']}

    output = {}

    val_waiting = [site_vars['mode'] == 'validation_waiting' for _, site_vars in input.items()]
    success = [site_vars.get('success', False) for _, site_vars in input.items()]
    train_waiting = [site_vars['mode'] == 'train_waiting' for _, site_vars in input.items()]
    test = [site_vars['mode'] == 'test' for _, site_vars in input.items()]
    train = [site_vars['mode'] == 'train' for _, site_vars in input.items()]

    if all(success):
        output['success'] = True
        output['test_prf1a'] = dict([(site, site_vars['test_prf1a']) for site, site_vars in input.items()])
        score = aggregate_scores(input, key='test_scores')
        cache['test_log'].append(score.prfa())
        cache['test_scores'] = [score.tp, score.tn, score.fp, score.fn]
        save_logs(cache, plot_keys=['validation_log'], file_keys=['test_log', 'test_scores'])

    elif all(val_waiting):
        for site, _ in input.items():
            sites_nn_state[site]['mode'] = 'validation'

    elif all(train_waiting):
        for site, site_vars in input.items():
            sites_nn_state[site]['mode'] = 'train'
        score = aggregate_scores(input, 'validation_scores')
        cache['validation_log'].append(score.prfa())
        if score.f1 > cache['best_val_score']:
            output['save_best_model'] = True
            cache.update(best_val_score=score.f1)

    elif any(train):
        avg_grads = aggregate_sites_grad(input, sites_nn_state)
        output['avg_grad_file'] = 'avg_grad.npy'
        np.save(state['transferDirectory'] + os.sep + output['avg_grad_file'], np.array(avg_grads))

    elif all(test):
        output['mode'] = 'test'

    output['global_nn_state'] = sites_nn_state

    output = json.dumps({'output': output, 'cache': cache, 'success': output.get('success', False)})
    sys.stdout.write(output)
