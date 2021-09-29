from coinstac_dinunet import COINNLocal, COINNRemote
from coinstac_dinunet.io import COINPyService

from nn_implementations.fs import FreeSurferDataset, FreeSurferTrainer, FSVDataHandle

TASK_FS = "FS-Classification"

""" Test """
task = TASK_FS
agg_engine = 'dSGD'


class Server(COINPyService):
    def get_local(self, msg) -> callable:
        pretrain_args = {'epochs': 21, 'batch_size': 16}
        dataloader_args = {"train": {"drop_last": True}}
        local = COINNLocal(task_id=task,
                           cache=self.cache, input=msg['data']['input'], batch_size=16,
                           state=msg['data']['state'], epochs=51, patience=21, model_scale=1,
                           pretrain_args=None,
                           dataloader_args=dataloader_args, agg_engine=agg_engine)

        if local.cache['task_id'] == TASK_FS:
            return local, FreeSurferTrainer, FreeSurferDataset, FSVDataHandle

    def get_remote(self, msg) -> callable:
        remote = COINNRemote(cache=self.cache, input=msg['data']['input'],
                             state=msg['data']['state'])

        if remote.cache['task_id'] == TASK_FS:
            return remote, FreeSurferTrainer


server = Server(verbose=False)
server.start()
