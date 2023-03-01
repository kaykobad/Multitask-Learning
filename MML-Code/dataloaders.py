import torch

from torch import nn
from torchvision import transforms
from torchvision.datasets import UCF101, HMDB51
from data_loading.kinetics_loader import KineticsDataset


class Datasets:
    ucf101 = "ucf101"
    hmdb51 = "hmdb51"
    kinetics400 = "kinetics400"


data_config = {
    "ucf101": {
        # "dir": "datasets\\data\\UCF-101",
        # "annotation": "datasets\\annotations\\UCF101-RecognitionTask",
        "dir": "datasets\\data\\Tiny-UCF",
        "annotation": "datasets\\annotations\\Tiny-UCF",
        "dataset": UCF101
    },
    "hmdb51": {
        "dir": "datasets\\data\\HMDB51",
        "annotation": "datasets\\annotations\\HMDB51",
        "dataset": HMDB51
    },
    "kinetics400": {
        "dir": "test-dataset/kinetics-400",
        "annotation": "test-dataset/kinetics-400/annotations.txt",
        "dataset": KineticsDataset
    }
}


def custom_collate(batch):
    filtered_batch = []
    for video, audio, label in batch:
        # print(audio.shape, video.shape, label)
        # audio = audio.mean(0)
        # print(audio.shape)
        # filtered_batch.append((video, audio, label))
        filtered_batch.append((video, audio, label))
        # print(video.shape, audio.shape, label)
    return torch.utils.data.dataloader.default_collate(filtered_batch)


def load_kinetics_dataset(batch_size=32, frame_per_clip=16, audio_sampling_rate=16000, audio_duration=10):
    config = data_config[Datasets.kinetics400]
    path = config["dir"]
    annotation = config["annotation"]
    dataset = config["dataset"]

    # Transforms
    v_tfs = transforms.Compose([
        # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
        # scale in [0, 1] of type float
        transforms.Lambda(lambda x: x / 255.),
        # reshape into (T, C, H, W) for easier convolutions
        transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        # rescale to the most common size
        transforms.Lambda(lambda x: nn.functional.interpolate(x, (224, 224))),
        # TODO: Uncomment the following 3 lines
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=32/255, saturation=0.4, contrast=0.4, hue=0.2),
        # transforms.ToTensor(),
    ])

    a_tfs = None

    # a_tfs = transforms.Compose([
    #     # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
    #     # scale in [0, 1] of type float
    #     transforms.Normalize(mean=0, std=1),
    #     # reshape into (T, C, H, W) for easier convolutions
    #     # transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
    #     # rescale to the most common size
    #     # transforms.Lambda(lambda x: nn.functional.interpolate(x, (224, 224))),
    #     # TODO: Uncomment the following 3 lines
    #     # transforms.RandomResizedCrop(224),
    #     # transforms.RandomHorizontalFlip(),
    #     # transforms.ColorJitter(brightness=32/255, saturation=0.4, contrast=0.4, hue=0.2),
    #     # transforms.ToTensor(),
    # ])

    train_dataset = dataset(path, annotation, frames_per_clip=frame_per_clip, audio_sampling_rate=audio_sampling_rate, audio_duration=audio_duration, audio_transforms=a_tfs, is_train=True, video_transforms=v_tfs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataset = dataset(path, annotation, frames_per_clip=frame_per_clip, audio_sampling_rate=audio_sampling_rate, audio_duration=audio_duration, audio_transforms=a_tfs, is_train=False, video_transforms=v_tfs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    print(f"Total number of train samples: {len(train_dataset)}")
    print(f"Total number of test samples: {len(test_dataset)}")
    print(f"Total number of (train) batches: {len(train_loader)}")
    print(f"Total number of (test) batches: {len(test_loader)}")
    print()

    # return test_dataset, test_loader
    return train_dataset, train_loader, test_dataset, test_loader


def load_dataset(name, batch_size=32, frame_per_clip=16):
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
        # transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        # rescale to the most common size
        transforms.Lambda(lambda x: nn.functional.interpolate(x, (224, 224))),
        # TODO: Uncomment the following 3 lines
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=32/255, saturation=0.4, contrast=0.4, hue=0.2),
        # transforms.ToTensor(),
    ])

    train_dataset = dataset(path, annotation, frames_per_clip=frame_per_clip, step_between_clips=1, train=True, transform=tfs, output_format='TCHW')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataset = dataset(path, annotation, frames_per_clip=frame_per_clip, step_between_clips=1, train=False, transform=tfs, output_format='TCHW')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    print(f"Total number of train samples: {len(train_dataset)}")
    print(f"Total number of test samples: {len(test_dataset)}")
    print(f"Total number of (train) batches: {len(train_loader)}")
    print(f"Total number of (test) batches: {len(test_loader)}")
    print()

    # return test_dataset, test_loader
    return train_dataset, train_loader, test_dataset, test_loader
