#!/usr/bin/python

from coinstac_dinunet import COINNRemote
from coinstac_dinunet.metrics import Prf1a
from coinstac_dinunet.io import RECV
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


if __name__ == "__main__":
    remote = FSRemote(cache=RECV['cache'], input=RECV['input'], state=RECV['state'])
    remote.compute()
    remote.send()
