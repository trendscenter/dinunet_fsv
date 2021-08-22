from coinstac_dinunet import COINNLocal, COINNRemote
from coinstac_dinunet.io import COINPyService
from nn_implementations.fcn import FreeSurferDataset, FreeSurferTrainer, FSVDataHandle
from nn_implementations.cnn3d import VBMDataset, VBMTrainer, VBMDataHandle

TASK_FSL = "FSL-Classification"
TASK_VBM = "VBM-Classification"


class Server(COINPyService):
    def get_local(self, msg) -> callable:
        pretrain_args = {'epochs': 11, 'batch_size': 16}
        local = COINNLocal(task_id=TASK_FSL,
                           cache=self.cache, input=msg['data']['input'],
                           pretrain_args=pretrain_args, batch_size=16,
                           state=msg['data']['state'], epochs=11, patience=21)

        if local.cache['task_id'] == TASK_FSL:
            return local, FreeSurferTrainer, FreeSurferDataset, FSVDataHandle
        elif local.cache['task_id'] == TASK_VBM:
            return local, VBMTrainer, VBMDataset, VBMDataHandle

    def get_remote(self, msg) -> callable:
        remote = COINNRemote(cache=self.cache, input=msg['data']['input'],
                             state=msg['data']['state'])

        if remote.cache['task_id'] == TASK_FSL:
            return remote, FreeSurferTrainer
        elif remote.cache['task_id'] == TASK_VBM:
            return remote, VBMTrainer


server = Server(verbose=False)
server.start()
