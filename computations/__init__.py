from enum import Enum

from .fs import *


class NNComputation(str, Enum):
    """ Available tasks """
    TASK_FREE_SURFER = "FS-Classification"


class AggEngine(str, Enum):
    DECENTRALIZED_SGD = "dSGD"
    RANK_DAD = "rankDAD"
