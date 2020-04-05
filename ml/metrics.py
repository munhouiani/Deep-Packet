from pathlib import Path

import numpy as np
import torch
import pandas as pd
from petastorm import make_reader
from petastorm.pytorch import DataLoader
from torch.nn import functional as F


def confusion_matrix(data_path, model, num_class):
    data_path = Path(data_path)
    model.eval()

    cm = np.zeros((num_class, num_class), dtype=np.float)

    dataloader = DataLoader(make_reader(str(data_path.absolute().as_uri()), reader_pool_type='process', num_epochs=1),
                            batch_size=4096)
    for batch in dataloader:
        x = batch['feature'].float()
        y = batch['label'].long()
        y_hat = torch.argmax(F.log_softmax(model(x), dim=1), dim=1)

        for i in range(len(y)):
            cm[y[i], y_hat[i]] += 1

    return cm


def get_precision(cm, i):
    tp = cm[i, i]
    tp_fp = cm[:, i].sum()

    return tp / tp_fp


def get_recall(cm, i):
    tp = cm[i, i]
    p = cm[i, :].sum()

    return tp / p


def get_classification_report(cm, labels=None):
    rows = []
    for i in range(cm.shape[0]):
        precision = get_precision(cm, i)
        recall = get_recall(cm, i)
        if labels:
            label = labels[i]
        else:
            label = i

        row = {
            'label': label,
            'precision': precision,
            'recall': recall
        }
        rows.append(row)

    return pd.DataFrame(rows)
