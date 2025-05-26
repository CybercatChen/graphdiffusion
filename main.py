import warnings
import os
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.loggers import TensorBoardLogger
from datasets.cow_dataset import CoWGraphDataModule, CoWDatasetInfos
from diffusion_model import FullDenoisingDiffusion

warnings.filterwarnings("ignore", category=PossibleUserWarning)


@hydra.main(version_base='1.3', config_path='configs', config_name='config_march')
def main(cfg: omegaconf.DictConfig):
    pl.seed_everything(cfg.train.seed)

    datamodule = CoWGraphDataModule(cfg)
    dataset_infos = CoWDatasetInfos(datamodule=datamodule, cfg=cfg)

    model = FullDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/",
                                          filename='{epoch}',
                                          monitor='val/epoch_NLL',
                                          save_top_k=5,
                                          mode='min',
                                          every_n_epochs=200)
    last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/", filename='last', every_n_epochs=200)
    callbacks.append(checkpoint_callback)
    callbacks.append(last_ckpt_save)
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=cfg.general.name)
    trainer = Trainer(logger=logger,
                      gradient_clip_val=cfg.train.clip_grad,
                      strategy='auto',
                      accelerator='gpu',
                      devices=[0],
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      callbacks=callbacks,
                      log_every_n_steps=50)

    if not cfg.general.test_only:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        trainer.test(model=model, datamodule=datamodule)
    else:
        for i in range(cfg.general.num_final_sampling):
            trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)


if __name__ == '__main__':
    main()
