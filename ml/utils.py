from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from ml.model import CNN, ResNet


def train_cnn(c1_kernel_size, c1_output_dim, c1_stride, c2_kernel_size, c2_output_dim, c2_stride, output_dim, data_path,
              epoch, model_path, signal_length, logger):
    # prepare dir for model path
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # seed everything
    seed_everything(seed=9876, workers=True)

    model = CNN(
        c1_kernel_size=c1_kernel_size,
        c1_output_dim=c1_output_dim,
        c1_stride=c1_stride,
        c2_kernel_size=c2_kernel_size,
        c2_output_dim=c2_output_dim,
        c2_stride=c2_stride,
        output_dim=output_dim,
        data_path=data_path,
        signal_length=signal_length,
    ).float()
    trainer = Trainer(val_check_interval=1.0, max_epochs=epoch, devices='auto', accelerator='auto', logger=logger,
                      callbacks=[EarlyStopping(monitor='training_loss', mode='min', check_on_train_epoch_end=True)])
    trainer.fit(model)

    # save model
    trainer.save_checkpoint(str(model_path.absolute()))


def train_resnet(c1_kernel_size, c1_output_dim, c1_stride, c1_groups, c1_n_block, output_dim, data_path,
              epoch, model_path, signal_length, logger):
    # prepare dir for model path
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # seed everything
    seed_everything(seed=9876, workers=True)

    model = ResNet(
        c1_kernel_size=c1_kernel_size,
        c1_output_dim=c1_output_dim,
        c1_stride=c1_stride,
        c1_groups=c1_groups,
        c1_n_block=c1_n_block,
        output_dim=output_dim,
        data_path=data_path,
        signal_length=signal_length,
    ).float()
    trainer = Trainer(val_check_interval=1.0, max_epochs=epoch, devices='auto', accelerator='auto', logger=logger,
                      callbacks=[EarlyStopping(monitor='training_loss', mode='min', check_on_train_epoch_end=True)])
    trainer.fit(model)

    # save model
    trainer.save_checkpoint(str(model_path.absolute()))


def train_application_classification_cnn_model(data_path, model_path):
    logger = TensorBoardLogger('application_classification_cnn_logs', 'application_classification_cnn')
    train_cnn(c1_kernel_size=4, c1_output_dim=200, c1_stride=3, c2_kernel_size=5, c2_output_dim=200, c2_stride=1,
              output_dim=17, data_path=data_path, epoch=20, model_path=model_path, signal_length=1500, logger=logger)


def train_application_classification_resnet_model(data_path, model_path):
    logger = TensorBoardLogger('application_classification_resnet_logs', 'application_classification_resnet')
    train_resnet(c1_kernel_size=4, c1_output_dim=16, c1_stride=3, c1_groups=1, c1_n_block=4, output_dim=17, data_path=data_path, epoch=40, model_path=model_path, signal_length=1500, logger=logger)


def train_traffic_classification_cnn_model(data_path, model_path):
    logger = TensorBoardLogger('traffic_classification_cnn_logs', 'traffic_classification_cnn')
    train_cnn(c1_kernel_size=5, c1_output_dim=200, c1_stride=3, c2_kernel_size=4, c2_output_dim=200, c2_stride=3,
              output_dim=12, data_path=data_path, epoch=20, model_path=model_path, signal_length=1500, logger=logger)


def train_traffic_classification_resnet_model(data_path, model_path):
    logger = TensorBoardLogger('traffic_classification_resnet_logs', 'traffic_classification_resnet')
    train_resnet(c1_kernel_size=5, c1_output_dim=16, c1_stride=3, c1_groups=1, c1_n_block=4, output_dim=12, data_path=data_path, epoch=40, model_path=model_path, signal_length=1500, logger=logger)


def load_cnn_model(model_path, gpu):
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    model = CNN.load_from_checkpoint(str(Path(model_path).absolute()), map_location=torch.device(device)).float().to(
        device)

    model.eval()

    return model


def load_resnet_model(model_path, gpu):
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    model = ResNet.load_from_checkpoint(str(Path(model_path).absolute()), map_location=torch.device(device)).float().to(
        device)

    model.eval()

    return model


def load_application_classification_cnn_model(model_path, gpu=False):
    return load_cnn_model(model_path=model_path, gpu=gpu)


def load_application_classification_resnet_model(model_path, gpu=False):
    return load_resnet_model(model_path=model_path, gpu=gpu)


def load_traffic_classification_cnn_model(model_path, gpu=False):
    return load_cnn_model(model_path=model_path, gpu=gpu)


def load_traffic_classification_resnet_model(model_path, gpu=False):
    return load_resnet_model(model_path=model_path, gpu=gpu)


def normalise_cm(cm):
    with np.errstate(all='ignore'):
        normalised_cm = cm / cm.sum(axis=1, keepdims=True)
        normalised_cm = np.nan_to_num(normalised_cm)
        return normalised_cm
