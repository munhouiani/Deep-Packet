from pathlib import Path

import torch
from petastorm import make_reader
from petastorm.pytorch import DataLoader
from pytorch_lightning import LightningModule
from torch import nn as nn
from torch.nn import functional as F


class CNN(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # config
        self.hparams = hparams
        self.data_path = self.hparams.data_path

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
        reader = make_reader(Path(self.data_path).absolute().as_uri(), reader_pool_type='process', workers_count=12,
                             pyarrow_serialize=True, shuffle_row_groups=True, shuffle_row_drop_partitions=2,
                             num_epochs=self.hparams.epoch)
        dataloader = DataLoader(reader, batch_size=16, shuffling_queue_capacity=4096)

        return dataloader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x = batch['feature'].float()
        y = batch['label'].long()
        y_hat = self(x)

        loss = {'loss': F.cross_entropy(y_hat, y)}

        if (batch_idx % 50) == 0:
            self.logger.log_metrics(loss, step=batch_idx)
        return loss
