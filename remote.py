import multiprocessing as mp

from coinstac_dinunet import COINNRemote

from computations import NNComputation, FreeSurferTrainer

_cache = {}
_pool = None


def args(cache):
    if cache['task_id'] == NNComputation.TASK_FREE_SURFER:
        return FreeSurferTrainer,
    else:
        raise ValueError(f"Invalid remote task:{cache.get('task')}")


def run(data):
    global _pool
    global _cache

    if _pool is None:
        _pool = mp.Pool(processes=data['input'].get('num_reducers', 2))

    remote = COINNRemote(
        cache=_cache, input=data['input'], state=data['state']
    )

    return remote(_pool, *args(_cache))
