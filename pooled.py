from coinstac_dinunet import PooledTrainer

from local import FreeSurferTrainer, FreeSurferDataset

if __name__ == "__main__":
    trainer = PooledTrainer(FreeSurferTrainer,
                            mode='train',
                            # pretrained_path='net_logs/weights.tar',
                            epochs=51, batch_size=32, patience=11,
                            learning_rate=0.001)
    trainer.run(FreeSurferDataset, only_sites=[0, 1, 2, 3, 4], only_folds=[0])
