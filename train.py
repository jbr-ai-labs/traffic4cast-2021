# coding: utf-8
__author__ = "sevakon: https://github.com/sevakon"

import torch
import datetime
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateLogger,
    ModelCheckpoint,
)

from traffic4cast.competition.competition_constants import (
    TRAIN_CITIES,
    CORE_CITES,
    EXTENDED_CITIES
)

from traffic4cast.pl_module import (
    T4CastBasePipeline,
    T4CastCorePipeline,
    DomainAdaptationBasePipeline,
    DomainAdaptationCorePipeline,
)

from traffic4cast.util.lightning_util import load_state_dict_from_lightning_checkpoint_

SEED = 111
seed_everything(SEED)


def main(hparams: Namespace):
    now = datetime.datetime.now().strftime("%m%d_%H:%M")
    experiment_name = f"{hparams.city}_{now}_{hparams.net}_{hparams.criterion}"

    wandb_logger = loggers.WandbLogger(
        name=experiment_name, save_dir="logs/",
        project="traffic4cast-2021", entity="sevakon")

    callbacks = [
        LearningRateLogger(), ]

    checkpoint_callback = ModelCheckpoint(
        filepath=f"weights/{experiment_name}_" +
        "best_{val_loss_2019:.4f}",
        monitor="val_loss_2019",
        save_top_k=10,
        mode="min",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(monitor="val_loss_2019",
                                        patience=13,
                                        mode="min",
                                        verbose=True)

    hparams.track_grad_norm = 2
    print(hparams)

    if "normal" == hparams.mode:
        lightning_module = T4CastCorePipeline(hparams=hparams) \
            if hparams.city in CORE_CITES \
            else T4CastBasePipeline(hparams=hparams)

    elif "domainadapt" == hparams.mode:
        if hparams.city in CORE_CITES:
            lightning_module = DomainAdaptationCorePipeline(hparams=hparams)
        else:
            lightning_module = DomainAdaptationBasePipeline(hparams=hparams)
            raise NotImplementedError()

    else:
        raise NotImplementedError("Other training modes are not implemented.")

    if hparams.model_checkpoint:
        if "domainadapt" == hparams.mode:
            load_state_dict_from_lightning_checkpoint_(
                lightning_module.net.model, hparams.model_checkpoint)
        else:
            load_state_dict_from_lightning_checkpoint_(
                lightning_module.net, hparams.model_checkpoint)

    hparams.logger = wandb_logger
    hparams.callbacks = callbacks
    hparams.checkpoint_callback = checkpoint_callback
    hparams.early_stop_callback = early_stop_callback

    trainer = Trainer.from_argparse_args(hparams)

    trainer.fit(lightning_module)

    # to make submission without lightning
    torch.save(lightning_module.net.state_dict(), f"weights/{experiment_name}.pth")


if __name__ == "__main__":
    # TODO: move configuration to *.yaml with Hydra
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--mode", choices=["normal", "domainadapt"],
                        default="normal", type=str)
    parser.add_argument("--city",
                        choices=TRAIN_CITIES + CORE_CITES + EXTENDED_CITIES,
                        type=str, required=True)
    parser.add_argument("--model_checkpoint", default=None, type=str)
    parser.add_argument("--city_static_map_path", default=None, type=str)
    parser.add_argument("--n_splits", default=None, type=int)
    parser.add_argument("--val_fold_idx", default=None, type=int)

    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--warmup_factor", default=1.0, type=int)

    parser.add_argument("--net", default="vanilla_unet", type=str, choices=[
        "vanilla_unet", "unet2020", "fitvid", "unet+rnn",
        "transformer", "naive_repeat_last", "resnext_unet",
        "densenet_unet", "efficientnetb3_unet",  "efficientnetb5_unet",
    ], )
    parser.add_argument("--emb_dim", default=1024 * 15 * 14, type=int)
    parser.add_argument("--criterion", choices=["mse", "ce+mse"], default="mse", type=str)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--scheduler", default="plateau", type=str)

    parser.add_argument("--sgd_momentum", default=0.9, type=float)
    parser.add_argument("--sgd_wd", default=1e-4, type=float)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    args = parser.parse_args()
    main(args)
