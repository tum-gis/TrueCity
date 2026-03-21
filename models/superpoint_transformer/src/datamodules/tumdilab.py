import logging
from src.datamodules.base import BaseDataModule
from src.datasets.tumdilab import Tumdilab, MiniTumdilab

log = logging.getLogger(__name__)

class TumdilabDataModule(BaseDataModule):
    """LightningDataModule for Tumdilab dataset.

    Keep this minimal like ScanNetDataModule:
    - Do not override `setup`;
    - Let BaseDataModule handle prepare/setup and transform wiring;
    - Dataset-specific options are accepted by the Dataset constructor.
    """
    _DATASET_CLASS = Tumdilab
    _MINIDATASET_CLASS = MiniTumdilab

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(
        root + "/configs/datamodule/semantic/tumdilab.yaml"
    )
    cfg.data_dir = root + "/data"
    _ = hydra.utils.instantiate(cfg)
