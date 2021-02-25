from coinstac_dinunet import PooledTrainer

from local import FreeSurferTrainer, FreeSurferDataset

if __name__ == "__main__":
    trainer = PooledTrainer(FreeSurferTrainer,
                            mode='train', log_dir='pooled_logs',
                            # pretrained_path='net_logs/weights.tar',
                            epochs=111, batch_size=32, patience=21, num_workers=8,
                            learning_rate=0.001, validation_epochs=2)
    trainer.run(FreeSurferDataset, only_sites=None, only_folds=None)
