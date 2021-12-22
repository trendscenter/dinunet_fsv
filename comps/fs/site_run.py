from coinstac_dinunet.site_runner import SiteRunner
from . import FreeSurferTrainer, FSVDataHandle, FreeSurferDataset

if __name__ == "__main__":
    runner = SiteRunner(taks_id='FSL', data_path='../../datasets/test_fsl', mode='Train', split_ratio=[0.8, 0.1, 0.1])
    runner.run(FreeSurferTrainer, FreeSurferDataset, FSVDataHandle)
