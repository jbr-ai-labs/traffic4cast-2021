# coding: utf-8
__author__ = "sevakon: https://github.com/sevakon"

from functools import partial
from typing import Optional
from typing import Callable
from typing import Union
from typing import Dict
from typing import List

import torch.nn.functional as F
import torch

from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, ReduceLROnPlateau
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from traffic4cast.models.baseline_unet import UNet, UNetTransfomer
from traffic4cast.util.warmup import GradualWarmupScheduler
from traffic4cast.data.dataset import T4CDataset


class T4CastBasePipeline(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self._city = hparams.city
        self.hparams = hparams

        self.net = self.get_net()
        self.criterion = self.get_criterion()

    def forward(self, x: torch.tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        train_step = {
            "loss": loss,
            "log": {
                f"train/{self.hparams.criterion}": loss
            },
        }

        return train_step

    def training_epoch_end(
        self,
        outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        train_epoch_end = {
            "train_loss": avg_loss,
            "log": {
                f"train/avg_{self.hparams.criterion}": avg_loss,
            },
        }

        return train_epoch_end

    def validation_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        #TODO: add masked MSE Loss calculation

        val_step = {
            "val_loss": loss,
            "val_masked_loss": None
        }

        return val_step

    def validation_epoch_end(
        self,
        outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        val_epoch_end = {
            "val_loss": avg_loss,
            "log": {
                f"val/avg_{self.hparams.criterion}": avg_loss,
            },
        }

        return val_epoch_end

    def get_net(self) -> Optional[torch.nn.Module]:
        if "vanilla_unet" == self.hparams.net:
            return UNet(
                in_channels=12 * 8,
                n_classes=6 * 7,
                depth=5,
                wf=6,
                padding=True,
                up_mode="upconv",
                batch_norm=True
            )
        elif "unet2020" == self.hparams.net:
            raise NotImplementedError()
        elif "fitvid" == self.hparams.net:
            raise NotImplementedError()
        elif "unet+rnn" == self.hparams.net:
            raise NotImplementedError()
        elif "transformer" == self.hparams.net:
            raise NotImplementedError()

    def get_criterion(self) -> Optional[Callable]:
        if "mse" == self.hparams.criterion:
            return F.mse_loss
        elif "ce+mse" == self.hparams.criterion:
            raise NotImplementedError("CrossEntropy + MSE loss not implemented")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        train_dataset = T4CDataset(
            root_dir=self.hparams.dataset_path,
            file_filter=f"{self._city}/training/2019*8ch.h5",
            transform=partial(UNetTransfomer.unet_pre_transform,
                              stack_channels_on_time=True,
                              zeropad2d=(6, 6, 1, 0), batch_dim=False)
        )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(train_dataset),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        val_dataset = T4CDataset(
            root_dir=self.hparams.dataset_path,
            file_filter=f"{self._city}/training/2020*8ch.h5",
            transform=partial(UNetTransfomer.unet_pre_transform,
                              stack_channels_on_time=True,
                              zeropad2d=(6, 6, 1, 0), batch_dim=False)
        )

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(val_dataset),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        return [optimizer], [scheduler]

    def get_optimizer(self) -> object:
        if "adam" == self.hparams.optimizer:
            return torch.optim.Adam(self.net.parameters(),
                                    lr=self.learning_rate)
        elif "adamw" == self.hparams.optimizer:
            return torch.optim.AdamW(self.net.parameters(),
                                     lr=self.learning_rate)
        elif "sgd" == self.hparams.optimizer:
            return torch.optim.SGD(
                self.net.parameters(),
                lr=self.learning_rate,
                momentum=self.hparams.sgd_momentum,
                weight_decay=self.hparams.sgd_wd,
            )
        else:
            raise NotImplementedError("Not a valid optimizer configuration.")

    def get_scheduler(self, optimizer) -> object:
        if "plateau" == self.hparams.scheduler:
            return ReduceLROnPlateau(optimizer)
        elif "plateau+warmup" == self.hparams.scheduler:
            plateau = ReduceLROnPlateau(optimizer)
            return GradualWarmupScheduler(
                optimizer,
                multiplier=self.hparams.warmup_factor,
                total_epoch=self.hparams.warmup_epochs,
                after_scheduler=plateau,
            )
        elif "cyclic" == self.hparams.scheduler:
            return CyclicLR(
                optimizer,
                base_lr=self.learning_rate / 100,
                max_lr=self.learning_rate,
                step_size_up=4000 / self.batch_size,
            )
        elif "cosine" == self.hparams.scheduler:
            return CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        elif "cosine+warmup" == self.hparams.scheduler:
            cosine = CosineAnnealingLR(
                optimizer,
                self.hparams.max_epochs - self.hparams.warmup_epochs)
            return GradualWarmupScheduler(
                optimizer,
                multiplier=self.hparams.warmup_factor,
                total_epoch=self.hparams.warmup_epochs,
                after_scheduler=cosine,
            )
        else:
            raise NotImplementedError("Not a valid scheduler configuration.")


class DomainAdaptationPipeline(T4CastBasePipeline):
    pass
