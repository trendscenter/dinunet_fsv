from coinstac_dinunet import COINNLocal, COINNRemote
from coinstac_dinunet.io import COINPyService

from nn_implementations.fs import FreeSurferDataset, FreeSurferTrainer, FSVDataHandle
from nn_implementations.vbm import VBMDataset, VBMTrainer, VBMDataHandle

TASK_FS = "FS-Classification"
TASK_VBM = "VBM-Classification"

""" Test """
task = TASK_FS
agg_engine = 'dSGD'


class Server(COINPyService):
    def get_local(self, msg) -> callable:
        pretrain_args = {'epochs': 11, 'batch_size': 16}
        dataloader_args = {"train": {"drop_last": True}}
        local = COINNLocal(task_id=task,
                           cache=self.cache, input=msg['data']['input'], batch_size=16,
                           state=msg['data']['state'], epochs=31, patience=21, model_scale=1,
                           pretrain_args=None,
                           dataloader_args=dataloader_args, agg_engine=agg_engine)

        if local.cache['task_id'] == TASK_FS:
            return local, FreeSurferTrainer, FreeSurferDataset, FSVDataHandle
        elif local.cache['task_id'] == TASK_VBM:
            return local, VBMTrainer, VBMDataset, VBMDataHandle

    def get_remote(self, msg) -> callable:
        remote = COINNRemote(cache=self.cache, input=msg['data']['input'],
                             state=msg['data']['state'])

        if remote.cache['task_id'] == TASK_FS:
            return remote, FreeSurferTrainer
        elif remote.cache['task_id'] == TASK_VBM:
            return remote, VBMTrainer


server = Server(verbose=False)
server.start()
