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
from traffic4cast.metrics.masking import get_static_mask
from traffic4cast.util.warmup import GradualWarmupScheduler
from traffic4cast.util.h5_util import load_h5_file
from traffic4cast.data.dataset import T4CDataset


class T4CastBasePipeline(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.net = self.get_net()
        self.criterion = self.get_criterion()

        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size

        self._city = hparams.city
        self._city_static_map: torch.Tensor = self._get_city_static_map()

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

        val_loss = self.criterion(y_hat, y)
        val_masked_loss = self.criterion(
            y_hat * self._city_static_map, y * self._city_static_map)

        mse_loss_by_sample = torch.mean(
            F.mse_loss(y_hat, y, reduction='none'), dim=(1, 2, 3))

        masked_mse_loss_by_sample = torch.mean(
            F.mse_loss(
                y_hat * self._city_static_map,
                y * self._city_static_map,
                reduction='none'),
            dim=(1, 2, 3)
        )

        normed_masked_mse_loss_by_sample = \
            masked_mse_loss_by_sample \
            * self._city_static_map.size().numel() \
            / torch.count_nonzero(self._city_static_map)

        val_step = {
            "loss": val_loss,
            "masked_loss": val_masked_loss,
            "mse_loss_by_sample": mse_loss_by_sample,
            "masked_mse_loss_by_sample": masked_mse_loss_by_sample,
            "normed_masked_mse_loss_by_sample": normed_masked_mse_loss_by_sample
        }

        return val_step

    def validation_epoch_end(
        self,
        outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_masked_loss = torch.stack(
            [x["masked_loss"] for x in outputs]).mean()
        avg_fair_mse_loss = torch.stack(
            [x["mse_loss_by_sample"] for x in outputs]).mean()
        avg_fair_masked_mse_loss = torch.stack(
            [x["masked_mse_loss_by_sample"] for x in outputs]).mean()
        avg_fair_normed_masked_mse_loss = torch.stack(
            [x["normed_masked_mse_loss_by_sample"] for x in outputs]).mean()

        val_epoch_end = {
            "val_loss": avg_loss,
            "val_masked_loss": avg_masked_loss,
            "val_fair_mse_loss": avg_fair_mse_loss,
            "val_fair_masked_mse_loss": avg_fair_masked_mse_loss,
            "val_fair_normed_masked_mse_loss": avg_fair_normed_masked_mse_loss,
            "log": {
                f"val/avg_{self.hparams.criterion}": avg_loss,
                f"val/avg_masked_{self.hparams.criterion}": avg_masked_loss,
                f"val/avg_fair_mse_loss": avg_fair_mse_loss,
                f"val/avg_fair_masked_mse_loss": avg_fair_masked_mse_loss,
                f"val/avg_fair_normed_masked_mse_loss": avg_fair_normed_masked_mse_loss,
            },
        }

        return val_epoch_end

    def get_net(self) -> Optional[torch.nn.Module]:
        if "vanilla_unet" == self.hparams.net:
            return UNet(
                in_channels=12 * 8,
                n_classes=6 * 8,
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

    def _get_city_static_map(self) -> torch.Tensor:
        mask = load_h5_file(self.hparams.city_static_map_path) \
            if self.hparams.city_static_map_path is not None \
            else get_static_mask(self._city, self.hparams.dataset_path)

        mask_torch = torch.from_numpy(mask)
        mask_torch_reshaped = torch.moveaxis(mask_torch, 2, 0)

        mask_torch_unsqueezed = torch.unsqueeze(mask_torch_reshaped, 0)

        zeropad2d = (6, 6, 1, 0) # TODO: move to config
        if zeropad2d is not None:
            padding = torch.nn.ZeroPad2d(zeropad2d)
            mask_torch_unsqueezed = padding(mask_torch_unsqueezed)

        summed_mask = torch.sum(mask_torch_unsqueezed[0], dim=0)
        mask_2d = torch.where(summed_mask > 0, 1, 0)

        return mask_2d

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        return [optimizer], [scheduler]

    def get_optimizer(self) -> torch.optim.Optimizer:
        if "adam" == self.hparams.optimizer:
            optimizer = torch.optim.Adam(self.net.parameters(),
                                    lr=self.learning_rate)
        elif "adamw" == self.hparams.optimizer:
            optimizer = torch.optim.AdamW(self.net.parameters(),
                                     lr=self.learning_rate)
        elif "sgd" == self.hparams.optimizer:
            optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.learning_rate,
                momentum=self.hparams.sgd_momentum,
                weight_decay=self.hparams.sgd_wd,
            )
        else:
            raise NotImplementedError("Not a valid optimizer configuration.")

        return optimizer

    def get_scheduler(self, optimizer) -> Union[
        ReduceLROnPlateau, CyclicLR, CosineAnnealingLR, GradualWarmupScheduler
    ]:
        if "plateau" == self.hparams.scheduler:
            scheduler = ReduceLROnPlateau(optimizer)
        elif "cyclic" == self.hparams.scheduler:
            scheduler = CyclicLR(
                optimizer,
                base_lr=self.learning_rate / 100,
                max_lr=self.learning_rate,
                step_size_up=4000 / self.batch_size,
            )
        elif "cosine" == self.hparams.scheduler:
            scheduler = CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        elif "cosine+warmup" == self.hparams.scheduler:
            cosine = CosineAnnealingLR(
                optimizer,
                self.hparams.max_epochs - self.hparams.warmup_epochs)
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=self.hparams.warmup_factor,
                total_epoch=self.hparams.warmup_epochs,
                after_scheduler=cosine,
            )
        else:
            raise NotImplementedError("Not a valid scheduler configuration.")

        return scheduler


class DomainAdaptationPipeline(T4CastBasePipeline):
    pass
