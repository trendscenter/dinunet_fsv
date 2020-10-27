#!/usr/bin/python
import json
import os
import shutil
import sys
from os import sep

import torch
from coinstac_pyprofiler import custom_profiler as cprof

from core import constants as cs
from core import utils as ut
from core.datasets import FreeSurferDataset
from core.nn import train_n_eval, init_dataset
from core.utils import init_k_folds, initialize_weights
from models import MSANNet

# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)

input_args = json.loads(sys.stdin.read())


def init_nn(cache, state, init_weights=False):
    """
    Initialize neural network/optimizer with locked parameters(check on remote script for that).
    Also detect and assign specified GPUs.
    @note Works only with one GPU per site at the moment.
    """
    device = ut.get_cuda_device(cache, state)
    model = MSANNet(in_size=cache['input_size'], hidden_sizes=cache['hidden_sizes'],
                    out_size=cache['num_class'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cache['learning_rate'])
    if init_weights:
        torch.manual_seed(cache['seed'])
        initialize_weights(model)
    return {'device': device, 'model': model.to(device), 'optimizer': optimizer}


@cprof.profile(type="pyinstrument",
               output_file_prefix=ut.get_output_file_path_and_prefix(input_args, cs.profile_log_dir_name))
def start_computation(args):
    cache = args['cache']
    input = args['input']
    state = args['state']
    out = {}

    nxt_phase = input.get('phase', 'init_runs')
    if nxt_phase == 'init_runs':
        """
        Generate folds as specified.
        """
        cache.update(**input)
        out.update(**init_k_folds(cache, state))
        cache['_mode_'] = input['mode']
        out['mode'] = cache['mode']

    if nxt_phase == 'init_nn':
        """
        Initialize neural network/optimizer and GPUs
        """
        cache.update(**input['run'][state['clientId']], epoch=0, cursor=0, train_log=[])
        cache['split_file'] = cache['splits'][str(cache['split_ix'])]
        cache['log_dir'] = state['outputDirectory'] + sep + cache['id'] + sep + f"fold_{cache['split_ix']}"
        os.makedirs(cache['log_dir'], exist_ok=True)
        cache['current_nn_state'] = 'current.nn.pt'
        cache['best_nn_state'] = 'best.nn.pt'
        nn = init_nn(cache, state, init_weights=True)
        ut.save_checkpoint(cache, nn, id=cache['current_nn_state'])
        init_dataset(cache, state, FreeSurferDataset)
        nxt_phase = 'computation'

    if nxt_phase == 'computation':
        """
        Train/validation and test phases
        """
        nn = init_nn(cache, state, init_weights=False)
        out_, nxt_phase = train_n_eval(nn, cache, input, state, FreeSurferDataset, nxt_phase)
        out.update(**out_)

    elif nxt_phase == 'success':
        """
        This phase receives global scores from the aggregator.
        """
        shutil.copy(f"{state['baseDirectory']}{sep}{input['results_zip']}.zip",
                    f"{state['outputDirectory'] + sep + cache['id']}{sep}{input['results_zip']}.zip")

    out['phase'] = nxt_phase
    output = json.dumps({'output': out, 'cache': cache})
    sys.stdout.write(output)
    args, cache, input, state, out, output = [None] * 6

    return


if __name__ == "__main__":
    start_computation(input_args)
