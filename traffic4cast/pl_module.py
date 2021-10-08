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
import glob

from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, ReduceLROnPlateau
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from traffic4cast.models.domain_adaptation import DomainAdaptationModel
from traffic4cast.models.pretrained_unet import PretrainedEncoderUNet
from traffic4cast.models.baseline_unet import UNet, UNetTransfomer
from traffic4cast.models.naive import NaiveRepeatLast
from traffic4cast.metrics.masking import get_static_mask
from traffic4cast.util.warmup import GradualWarmupScheduler
from traffic4cast.util.h5_util import load_h5_file
from traffic4cast.data.dataset import T4CDataset, ExpandedDataset, ConcatDataset


class T4CastBasePipeline(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.net = self.get_net()
        self.criterion = self.get_criterion()

        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size

        self.padding = (6, 6, 1, 0)
        if self.hparams.net in ['densenet_unet', 'resnext_unet'] \
                or self.hparams.net.startswith("efficientnet"):
            self.padding = (6, 6, 9, 8) # input image 512 x 448

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

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: int
    ) -> Dict[str, torch.Tensor]:
        """ dataloader_idx = 0 -> val on 2019, dataloader_idx = 1 -> 2020 """
        if dataloader_idx == 0:
            year = 2019
        elif dataloader_idx == 1:
            year = 2020
        else:
            raise AssertionError(
                "Dataloader index expected to be within [0, 1] range")

        x, y = batch
        y_hat = self.forward(x)

        val_loss = self.criterion(y_hat, y).detach().item()
        val_masked_loss = self.criterion(
            y_hat * self._city_static_map.type_as(y_hat),
            y * self._city_static_map.type_as(y)
        ).detach().item()

        mse_loss_by_sample = torch.mean(
            F.mse_loss(y_hat, y, reduction='none'), dim=(1, 2, 3)).detach().cpu().numpy()

        masked_mse_loss_by_sample = torch.mean(
            F.mse_loss(
                y_hat * self._city_static_map.type_as(y_hat),
                y * self._city_static_map.type_as(y),
                reduction='none'),
            dim=(1, 2, 3)
        ).detach().cpu().numpy()

        # No need to saved normed metrics here since
        # it can be calculated as MEAN maskes loss multiplied by a ratio
        # normed_masked_mse_loss_by_sample = \
        # masked_mse_loss_by_sample \
        # * self._city_static_map.size().numel() \
        # / torch.count_nonzero(self._city_static_map)

        val_step = {
            f"loss": torch.tensor(val_loss),
            f"masked_loss": torch.tensor(val_masked_loss),
            f"mse_loss_by_sample": torch.tensor(mse_loss_by_sample),
            f"masked_mse_loss_by_sample": torch.tensor(masked_mse_loss_by_sample),
            # f"normed_masked_mse_loss_by_sample": normed_masked_mse_loss_by_sample
        }

        return val_step

    def validation_epoch_end(
        self,
        outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        # Validation Epoch Mean Loss
        avg_loss_2019 = torch.stack([x["loss"] for x in outputs[0]]).mean()
        avg_loss_2020 = torch.stack([x["loss"] for x in outputs[1]]).mean()

        # Validation Epoch Mean City-Masked Loss
        avg_masked_loss_2019 = torch.stack(
            [x["masked_loss"] for x in outputs[0]]).mean()
        avg_masked_loss_2020 = torch.stack(
            [x["masked_loss"] for x in outputs[1]]).mean()

        # Validation Epoch Mean L2 Loss
        avg_fair_mse_loss_2019 = torch.stack(
            [x["mse_loss_by_sample"] for x in outputs[0]]).mean()
        avg_fair_mse_loss_2020 = torch.stack(
            [x["mse_loss_by_sample"] for x in outputs[1]]).mean()

        # Validation Epoch Mean City-Masked L2 Loss
        avg_fair_masked_mse_loss_2019 = torch.stack(
            [x["masked_mse_loss_by_sample"] for x in outputs[0]]).mean()
        avg_fair_masked_mse_loss_2020 = torch.stack(
            [x["masked_mse_loss_by_sample"] for x in outputs[1]]).mean()

        # Validation Epoch Mean Normed City-Masked L2 Loss
        avg_fair_normed_masked_mse_loss_2019 = avg_fair_masked_mse_loss_2019 \
            * self._city_static_map.size().numel() \
            / torch.count_nonzero(self._city_static_map)
        avg_fair_normed_masked_mse_loss_2020 = avg_fair_masked_mse_loss_2020 \
            * self._city_static_map.size().numel() \
            / torch.count_nonzero(self._city_static_map)

        val_epoch_end = {
            "val_loss_2019": avg_loss_2019,
            "val_loss_2020": avg_loss_2020,
            "val_masked_loss_2019": avg_masked_loss_2019,
            "val_masked_loss_2020": avg_masked_loss_2020,
            "val_fair_mse_loss_2019": avg_fair_mse_loss_2019,
            "val_fair_mse_loss_2020": avg_fair_mse_loss_2020,
            "val_fair_masked_mse_loss_2019": avg_fair_masked_mse_loss_2019,
            "val_fair_masked_mse_loss_2020": avg_fair_masked_mse_loss_2020,
            "val_fair_normed_masked_mse_loss_2019": avg_fair_normed_masked_mse_loss_2019,
            "val_fair_normed_masked_mse_loss_2020": avg_fair_normed_masked_mse_loss_2020,
            "log": {
                f"val/avg_{self.hparams.criterion}/2019": avg_loss_2019,
                f"val/avg_{self.hparams.criterion}/2020": avg_loss_2020,
                f"val/avg_masked_{self.hparams.criterion}/2019": avg_masked_loss_2019,
                f"val/avg_masked_{self.hparams.criterion}/2020": avg_masked_loss_2020,
                f"val/avg_fair_mse_loss/2019": avg_fair_mse_loss_2019,
                f"val/avg_fair_mse_loss/2020": avg_fair_mse_loss_2020,
                f"val/avg_fair_masked_mse_loss/2019": avg_fair_masked_mse_loss_2019,
                f"val/avg_fair_masked_mse_loss/2020": avg_fair_masked_mse_loss_2020,
                f"val/avg_fair_normed_masked_mse_loss/2019": avg_fair_normed_masked_mse_loss_2019,
                f"val/avg_fair_normed_masked_mse_loss/2020": avg_fair_normed_masked_mse_loss_2020,
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
        elif "resnext_unet" == self.hparams.net:
            return PretrainedEncoderUNet(
                encoder='resnext50_32x4d',
                in_channels=12 * 8,
                n_classes=6 * 8,)
        elif "densenet_unet" == self.hparams.net:
            return PretrainedEncoderUNet(
                encoder='densenet201',
                in_channels=12 * 8,
                n_classes=6 * 8, depth=5)
        elif "efficientnetb3_unet" == self.hparams.net:
            return PretrainedEncoderUNet(
                encoder="efficientnet-b3",
                in_channels=12 * 8,
                n_classes=6 * 8, depth=6)
        elif "efficientnetb5_unet" == self.hparams.net:
            return PretrainedEncoderUNet(
                encoder="efficientnet-b5",
                in_channels=12 * 8,
                n_classes=6 * 8, depth=6)
        elif "unet2020" == self.hparams.net:
            raise NotImplementedError()
        elif "naive_repeat_last" == self.hparams.net:
            return NaiveRepeatLast()
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
                              zeropad2d=self.padding, batch_dim=False),
            n_splits=self.hparams.n_splits,
            folds_to_use=tuple(
                [i for i in range(self.hparams.n_splits)
                 if i != self.hparams.val_fold_idx])
            if self.hparams.n_splits is not None
            and self.hparams.val_fold_idx is not None else None
        )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(train_dataset),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> List[torch.utils.data.DataLoader]:
        """ Returns two dataloaders:
        first one to validate on the slice of 2019 data,
        the other one on 2020 data """
        val_2019_dataset = T4CDataset(
            root_dir=self.hparams.dataset_path,
            file_filter=f"{self._city}/training/2019*8ch.h5",
            transform=partial(UNetTransfomer.unet_pre_transform,
                              stack_channels_on_time=True,
                              zeropad2d=self.padding, batch_dim=False),
            n_splits=self.hparams.n_splits,
            folds_to_use=tuple([self.hparams.val_fold_idx])
            if self.hparams.val_fold_idx is not None else None,
        )

        val_2020_dataset = T4CDataset(
            root_dir=self.hparams.dataset_path,
            file_filter=f"{self._city}/training/2020*8ch.h5",
            transform=partial(UNetTransfomer.unet_pre_transform,
                              stack_channels_on_time=True,
                              zeropad2d=self.padding, batch_dim=False)
        )

        return [
            DataLoader(
                val_2019_dataset,
                batch_size=self.batch_size,
                sampler=SequentialSampler(val_2019_dataset),
                num_workers=self.hparams.num_workers,
                pin_memory=True,
            ),
            DataLoader(
                val_2020_dataset,
                batch_size=self.batch_size,
                sampler=SequentialSampler(val_2020_dataset),
                num_workers=self.hparams.num_workers,
                pin_memory=True,
            )
        ]

    def _get_city_static_map(self) -> torch.Tensor:
        mask = load_h5_file(self.hparams.city_static_map_path) \
            if self.hparams.city_static_map_path is not None \
            else get_static_mask(self._city, self.hparams.dataset_path)

        mask_torch = torch.from_numpy(mask)
        mask_torch_reshaped = torch.moveaxis(mask_torch, 2, 0)

        mask_torch_unsqueezed = torch.unsqueeze(mask_torch_reshaped, 0)

        zeropad2d = self.padding # TODO: move to config
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


class T4CastCorePipeline(T4CastBasePipeline):

    def validation_step(
        self, batch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return super().validation_step(batch, batch_idx, 0)

    def validation_epoch_end(
        self,
        outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        # Validation Epoch Mean Loss
        avg_loss_2019 = torch.stack([x["loss"] for x in outputs]).mean()

        # Validation Epoch Mean City-Masked Loss
        avg_masked_loss_2019 = torch.stack(
            [x["masked_loss"] for x in outputs]).mean()

        # Validation Epoch Mean L2 Loss
        avg_fair_mse_loss_2019 = torch.stack(
            [x["mse_loss_by_sample"] for x in outputs]).mean()

        # Validation Epoch Mean City-Masked L2 Loss
        avg_fair_masked_mse_loss_2019 = torch.stack(
            [x["masked_mse_loss_by_sample"] for x in outputs]).mean()

        # Validation Epoch Mean Normed City-Masked L2 Loss
        avg_fair_normed_masked_mse_loss_2019 = avg_fair_masked_mse_loss_2019 \
            * self._city_static_map.size().numel() \
            / torch.count_nonzero(self._city_static_map)

        val_epoch_end = {
            "val_loss_2019": avg_loss_2019,
            "val_masked_loss_2019": avg_masked_loss_2019,
            "val_fair_mse_loss_2019": avg_fair_mse_loss_2019,
            "val_fair_masked_mse_loss_2019": avg_fair_masked_mse_loss_2019,
            "val_fair_normed_masked_mse_loss_2019": avg_fair_normed_masked_mse_loss_2019,
            "log": {
                f"val/avg_{self.hparams.criterion}/2019": avg_loss_2019,
                f"val/avg_masked_{self.hparams.criterion}/2019": avg_masked_loss_2019,
                f"val/avg_fair_mse_loss/2019": avg_fair_mse_loss_2019,
                f"val/avg_fair_masked_mse_loss/2019": avg_fair_masked_mse_loss_2019,
                f"val/avg_fair_normed_masked_mse_loss/2019": avg_fair_normed_masked_mse_loss_2019,
            },
        }

        return val_epoch_end

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """ Returns one dataloader:
        one to validate on the slice of 2019 data """
        val_2019_dataset = T4CDataset(
            root_dir=self.hparams.dataset_path,
            file_filter=f"{self._city}/training/2019*8ch.h5",
            transform=partial(UNetTransfomer.unet_pre_transform,
                              stack_channels_on_time=True,
                              zeropad2d=self.padding, batch_dim=False),
            n_splits=self.hparams.n_splits,
            folds_to_use=tuple([self.hparams.val_fold_idx])
            if self.hparams.val_fold_idx is not None else None,
        )

        return DataLoader(
            val_2019_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(val_2019_dataset),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


class DomainAdaptationBasePipeline(T4CastBasePipeline):

    def __init__(self, hparams):
        super(DomainAdaptationBasePipeline, self).__init__(hparams)

        self.net = DomainAdaptationModel(self.net, self.hparams.emb_dim)


class DomainAdaptationCorePipeline(T4CastCorePipeline):

    def __init__(self, hparams):
        super(DomainAdaptationCorePipeline, self).__init__(hparams)

        self.net = DomainAdaptationModel(self.net, self.hparams.emb_dim)

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        (source_domain, y), target_domain = batch

        y_hat = self.forward(source_domain)
        loss = self.criterion(y_hat, y)

        net_on_source = self.net.predict_domain(source_domain)
        loss_on_source = F.binary_cross_entropy_with_logits(
            net_on_source, torch.zeros_like(net_on_source))

        net_on_target = self.net.predict_domain(target_domain)
        loss_on_target = F.binary_cross_entropy_with_logits(
            net_on_source, torch.ones_like(net_on_target))

        train_step = {
            "loss": loss + loss_on_source + loss_on_target,
            f"{self.hparams.criterion}": loss,
            "domain_classification_source_loss": loss_on_source,
            "domain_classification_target_loss": loss_on_target,
            "log": {
                f"train/{self.hparams.criterion}": loss,
                f"train/domain_classification_source_loss": loss_on_source,
                f"train/domain_classification_target_loss": loss_on_target,
            },
        }

        return train_step

    def training_epoch_end(
        self,
        outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        criterion = torch.stack([x[f"{self.hparams.criterion}"] for x in outputs]).mean()
        domain_classification_source_loss = \
            torch.stack([x["domain_classification_source_loss"] for x in outputs]).mean()
        domain_classification_target_loss = \
            torch.stack([x["domain_classification_target_loss"] for x in outputs]).mean()

        train_epoch_end = {
            "train_loss": avg_loss,
            "log": {
                f"train/avg_loss": avg_loss,
                f"train/avg_{self.hparams.criterion}": criterion,
                f"train/avg_domain_classification_source_loss": domain_classification_source_loss,
                f"train/avg_domain_classification_target_loss": domain_classification_target_loss,
            },
        }

        return train_epoch_end

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        train_dataset = T4CDataset(
            root_dir=self.hparams.dataset_path,
            file_filter=f"{self._city}/training/2019*8ch.h5",
            transform=partial(UNetTransfomer.unet_pre_transform,
                              stack_channels_on_time=True,
                              zeropad2d=self.padding, batch_dim=False),
            n_splits=self.hparams.n_splits,
            folds_to_use=tuple(
                [i for i in range(self.hparams.n_splits)
                 if i != self.hparams.val_fold_idx])
            if self.hparams.n_splits is not None
            and self.hparams.val_fold_idx is not None else None
        )

        competition_file = glob.glob(
            f"{self.hparams.dataset_path}/{self._city}/*test_temporal.h5",
            recursive=True)[0]

        test_data = load_h5_file(competition_file, to_torch=True)
        train_target_domain_dataset = ExpandedDataset(
            initial_dataset=torch.utils.data.TensorDataset(test_data),
            desired_length=len(train_dataset),
            transform=partial(UNetTransfomer.unet_pre_transform,
                              stack_channels_on_time=True,
                              zeropad2d=self.padding, batch_dim=False),
        )

        return DataLoader(
            ConcatDataset(
                train_dataset,
                train_target_domain_dataset
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
