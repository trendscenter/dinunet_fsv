from coinstac_dinunet import COINNLocal, COINNRemote
from coinstac_dinunet.io import COINPyService
from coinstac_dinunet.metrics import Prf1a
from classification import FreeSurferDataset, FreeSurferTrainer, FSVDataHandle


class FSRemote(COINNRemote):
    def _set_monitor_metric(self):
        self.cache['monitor_metric'] = 'f1', 'maximize'

    def _set_log_headers(self):
        self.cache['log_header'] = 'Loss|Accuracy,F1'

    def _new_metrics(self):
        return Prf1a()


class Server(COINPyService):

    def get_local(self, msg) -> callable:
        pretrain_args = {'epochs': 11, 'batch_size': 16}
        local = COINNLocal(cache=self.cache, input=msg['data']['input'],
                           pretrain_args=pretrain_args, batch_size=16,
                           state=msg['data']['state'], epochs=5, patience=21,
                           computation_id='coinstac_dinunet_fsl')
        return local

    def get_remote(self, msg) -> callable:
        remote = FSRemote(cache=self.cache, input=msg['data']['input'],
                          state=msg['data']['state'])
        return remote

    def get_local_compute_args(self, msg) -> list:
        return [FreeSurferTrainer, FreeSurferDataset, FSVDataHandle]


server = Server(verbose=False)
server.start()
