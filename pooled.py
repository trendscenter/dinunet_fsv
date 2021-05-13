
import coinstac_dinunet.io as io
io.enable_io = False

from coinstac_dinunet import PooledTrainer
from local import FreeSurferTrainer, FreeSurferDataset

if __name__ == "__main__":
    trainer = PooledTrainer(FreeSurferTrainer,
                            mode='train', log_dir='pooled_logs',
                            epochs=21, batch_size=32, patience=21, num_workers=8,
                            learning_rate=0.001, validation_epochs=1)
    trainer.run(FreeSurferDataset, only_sites=None, only_folds=[0])
