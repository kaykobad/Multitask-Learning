import torch
from dataloaders import load_dataset, Datasets

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


if __name__ == '__main__':
    test_dataset, test_dataloader = load_dataset(Datasets.ucf101)
    v, a, l = next(iter(test_dataloader))
    print(v.shape, a.shape, l.shape)
    #
    # print(video.shape, video[0].shape, video[1].shape)
    # print(audio.shape, audio[0].shape, audio[1].shape)
    # print(label.shape, label[0].shape, label[1].shape)
