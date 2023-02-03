import torch

from torch import nn
from torchvision import transforms
from torchvision.datasets import UCF101, HMDB51


class Datasets:
    ucf101 = "ucf101"
    hmdb51 = "hmdb51"


data_config = {
    "ucf101": {
        "dir": "datasets\\data\\UCF-101",
        "annotation": "datasets\\annotations\\UCF101-RecognitionTask",
        "dataset": UCF101
    },
    "hmdb51": {
        "dir": "datasets\\data\\HMDB51",
        "annotation": "datasets\\annotations\\HMDB51",
        "dataset": HMDB51
    }
}


def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


def load_dataset(name, batch_size=32, frame_per_clip=8):
    config = data_config[name]
    path = config["dir"]
    annotation = config["annotation"]
    dataset = config["dataset"]

    # Transforms
    tfs = transforms.Compose([
        # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
        # scale in [0, 1] of type float
        transforms.Lambda(lambda x: x / 255.),
        # reshape into (T, C, H, W) for easier convolutions
        transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        # rescale to the most common size
        transforms.Lambda(lambda x: nn.functional.interpolate(x, (224, 224))),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=32/255, saturation=0.4, contrast=0.4, hue=0.2),
        # transforms.ToTensor(),
    ])

    train_dataset = dataset(path, annotation, frames_per_clip=frame_per_clip, step_between_clips=1, train=True, transform=tfs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataset = dataset(path, annotation, frames_per_clip=frame_per_clip, step_between_clips=1, train=False, transform=tfs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    print(f"Total number of train samples: {len(train_dataset)}")
    print(f"Total number of test samples: {len(test_dataset)}")
    print(f"Total number of (train) batches: {len(train_loader)}")
    print(f"Total number of (test) batches: {len(test_loader)}")
    print()

    # return test_dataset, test_loader
    return train_dataset, train_loader, test_dataset, test_loader
