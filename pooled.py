from coinstac_dinunet import PooledTrainer

from local import FreeSurferTrainer, FreeSurferDataset

if __name__ == "__main__":
    trainer = PooledTrainer(FreeSurferTrainer,
                            mode='train',
                            # pretrained_path='net_logs/weights.tar',
                            epochs=111, batch_size=32, patience=21,
                            learning_rate=0.001, validation_epochs=2)
    trainer.run(FreeSurferDataset, only_sites=[0, 1, 2, 3, 4], only_folds=None)
