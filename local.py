import multiprocessing as mp

import coinstac
from coinstac_dinunet import COINNLocal

from comps import AggEngine
from comps import NNComputation, FreeSurferDataset, FreeSurferTrainer, FSVDataHandle

""" Test """
computation = NNComputation.TASK_FREE_SURFER
agg_engine = AggEngine.DECENTRALIZED_SGD

_cache = {}
_pool = None


def args(cache):
    if cache['task_id'] == NNComputation.TASK_FREE_SURFER:
        return FreeSurferTrainer, FreeSurferDataset, FSVDataHandle
    else:
        raise ValueError(f"Invalid local task:{cache.get('task')}")


def run(data):
    global _pool
    global _cache

    _cache['total_duration'] = coinstac.compTime
    if _pool is None:
        _pool = mp.Pool(processes=data['input'].get('num_reducers', 2))

    pretrain_args = {'epochs': 11, 'batch_size': 16}
    dataloader_args = {"train": {"drop_last": True}}
    local = COINNLocal(
        task_id=computation, agg_engine=agg_engine,
        cache=_cache, input=data['input'], batch_size=16,
        state=data['state'], epochs=15, patience=31, split_ratio=[0.7, 0.15, 0.15],
        pretrain_args=None, dataloader_args=dataloader_args
    )

    return local(_pool, *args(_cache))
