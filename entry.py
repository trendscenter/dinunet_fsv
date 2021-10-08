import time

import coinstac
from coinstac_dinunet.utils import duration

import local
import remote
start = time.time()
print("Start time: ", start)
coinstac.start(local.run, remote.run)
print("*** Total runtime *** ", duration(start))
