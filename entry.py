from coinstac_dinunet.io import COINPyService
from coinstac_dinunet import COINNLocal
from local import FreeSurferDataset, FreeSurferTrainer

from coinstac_dinunet import COINNRemote
from coinstac_dinunet.metrics import Prf1a


#
# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)


class FSRemote(COINNRemote):
    def _set_monitor_metric(self):
        self.cache['monitor_metric'] = 'f1', 'maximize'

    def _set_log_headers(self):
        self.cache['log_header'] = 'Loss|Accuracy,F1'

    def _new_metrics(self):
        return Prf1a()


class Server(COINPyService):

    def _local(self, msg) -> callable:
        pretrain_args = {'epochs': 51, 'batch_size': 16}
        local = COINNLocal(cache=self.cache, input=msg['data']['input'],
                           pretrain_args=None, batch_size=16,
                           state=msg['data']['state'], epochs=21, patience=21, computation_id='fsv_quick')
        return local

    def _remote(self, msg) -> callable:
        remote = FSRemote(cache=self.cache, input=msg['data']['input'],
                          state=msg['data']['state'])
        return remote

    def _local_compute_args(self, msg) -> list:
        return [FreeSurferDataset, FreeSurferTrainer]


server = Server()
server.start()
