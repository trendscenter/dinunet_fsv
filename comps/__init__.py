from enum import Enum

from .fs import *
from .icalstm import *


class NNComputation(str, Enum):
    """ Available tasks """
    TASK_FREE_SURFER = "FS-Classification"
    TASK_ICA = "ICA-Classification"


class AggEngine(str, Enum):
    DECENTRALIZED_SGD = "dSGD"
    RANK_DAD = "rankDAD"
