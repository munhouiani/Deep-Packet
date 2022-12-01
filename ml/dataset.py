import torch


def dataset_collate_function(batch):
    feature = torch.stack([torch.tensor([data["feature"]]) for data in batch])
    label = torch.tensor([data["label"] for data in batch])
    transformed_batch = {"feature": feature, "label": label}
    return transformed_batch
