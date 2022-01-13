import multiprocessing

import datasets
import torch
from pytorch_lightning import LightningModule
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml.dataset import dataset_collate_function


class CNN(LightningModule):
    def __init__(self, c1_output_dim, c1_kernel_size, c1_stride, c2_output_dim, c2_kernel_size, c2_stride,
                 output_dim, data_path, signal_length):
        super().__init__()
        # save parameters to checkpoint
        self.save_hyperparameters()

        # two convolution, then one max pool
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.hparams.c1_output_dim,
                kernel_size=self.hparams.c1_kernel_size,
                stride=self.hparams.c1_stride
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hparams.c1_output_dim,
                out_channels=self.hparams.c2_output_dim,
                kernel_size=self.hparams.c2_kernel_size,
                stride=self.hparams.c2_stride
            ),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool1d(
            kernel_size=2
        )

        # flatten, calculate the output size of max pool
        # use a dummy input to calculate
        dummy_x = torch.rand(1, 1, self.hparams.signal_length, requires_grad=False)
        dummy_x = self.conv1(dummy_x)
        dummy_x = self.conv2(dummy_x)
        dummy_x = self.max_pool(dummy_x)
        max_pool_out = dummy_x.view(1, -1).shape[1]

        # followed by 5 dense layers
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=max_pool_out,
                out_features=200
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=200,
                out_features=100
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(
                in_features=100,
                out_features=50
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )

        # finally, output layer
        self.out = nn.Linear(
            in_features=50,
            out_features=self.hparams.output_dim
        )

    def forward(self, x):
        # make sure the input is in [batch_size, channel, signal_length]
        # where channel is 1
        # signal_length is 1500 by default
        batch_size = x.shape[0]

        # 2 conv 1 max
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = x.reshape(batch_size, -1)

        # 3 fc
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # output
        x = self.out(x)

        return x

    def train_dataloader(self):
        # expect to get train folder
        dataset_dict = datasets.load_dataset(self.hparams.data_path)
        dataset = dataset_dict[list(dataset_dict.keys())[0]]
        try:
            num_workers = multiprocessing.cpu_count()
        except:
            num_workers = 1
        dataloader = DataLoader(dataset, batch_size=16, num_workers=num_workers, collate_fn=dataset_collate_function,
                                shuffle=True)

        return dataloader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x = batch['feature'].float()
        y = batch['label'].long()
        y_hat = self(x)

        entropy = F.cross_entropy(y_hat, y)
        self.log('training_loss', entropy, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        loss = {'loss': entropy}

        return loss
