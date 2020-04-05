from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from ml.model import CNN


def train_cnn(c1_kernel_size, c1_output_dim, c1_stride, c2_kernel_size, c2_output_dim, c2_stride, output_dim, data_path,
              epoch, gpus, model_path, signal_length, logger):
    # prepare dir for model path
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # build hparams
    hparams = Namespace(**{
        'c1_kernel_size': c1_kernel_size,
        'c1_output_dim': c1_output_dim,
        'c1_stride': c1_stride,
        'c2_kernel_size': c2_kernel_size,
        'c2_output_dim': c2_output_dim,
        'c2_stride': c2_stride,
        'output_dim': output_dim,
        'data_path': data_path,
        'signal_length': signal_length,
        'epoch': epoch
    })
    model = CNN(hparams).float()
    trainer = Trainer(val_check_interval=100, max_epochs=1, gpus=gpus, logger=logger)
    trainer.fit(model)

    # save model
    trainer.save_checkpoint(str(model_path.absolute()))


def train_application_classification_cnn_model(data_path, model_path, gpu):
    logger = TensorBoardLogger('application_classification_cnn_logs', 'application_classification_cnn')
    train_cnn(c1_kernel_size=4, c1_output_dim=200, c1_stride=3, c2_kernel_size=5, c2_output_dim=200, c2_stride=1,
              output_dim=17, data_path=data_path, epoch=300, gpus=gpu, model_path=model_path, signal_length=1500,
              logger=logger)


def train_traffic_classification_cnn_model(data_path, model_path, gpu):
    logger = TensorBoardLogger('traffic_classification_cnn_logs', 'traffic_classification_cnn')
    train_cnn(c1_kernel_size=5, c1_output_dim=200, c1_stride=3, c2_kernel_size=4, c2_output_dim=200, c2_stride=3,
              output_dim=12, data_path=data_path, epoch=300, gpus=gpu, model_path=model_path, signal_length=1500,
              logger=logger)


def load_cnn_model(model_path):
    model = CNN.load_from_checkpoint(str(Path(model_path).absolute()), map_location=torch.device('cpu')).float().to(
        'cpu')

    model.eval()

    return model


def load_application_classification_cnn_model(model_path):
    return load_cnn_model(model_path=model_path)


def load_traffic_classification_cnn_model(model_path):
    return load_cnn_model(model_path=model_path)


def normalise_cm(cm):
    with np.errstate(all='ignore'):
        normalised_cm = cm / cm.sum(axis=1, keepdims=True)
        normalised_cm = np.nan_to_num(normalised_cm)
        return normalised_cm
