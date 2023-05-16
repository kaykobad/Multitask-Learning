import torch
import torchvision
import torch.utils.data as data
from torch.utils.data import Dataset


class PbvsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.EO_dir = root + "EO/"
        self.SAR_dir = root + "SAR/"
        self.transform = transform
        self.EO_dataset = torchvision.datasets.ImageFolder(root=self.EO_dir, transform=transform)
        self.SAR_dataset = torchvision.datasets.ImageFolder(root=self.SAR_dir, transform=transform)
        self.targets = self.EO_dataset.targets
        # print(type(self.EO_dataset[1]), self.EO_dataset[1])
        # print(torch.equal(torch.as_tensor(self.EO_dataset.targets), torch.as_tensor(self.SAR_dataset.targets)))
        # for i in range(len(self.targets)):
        #     if self.EO_dataset.targets[i] != self.SAR_dataset.targets[i]:
        #         print("No match at " + str(i))

        # print("end")
        
    def __len__(self):
        return len(self.EO_dataset)

    def __getitem__(self, idx):
        return self.EO_dataset[idx][0], self.SAR_dataset[idx][0], self.targets[idx]


class PbvsTestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.EO_dir = root + "EO/"
        self.SAR_dir = root + "SAR/"
        self.transform = transform
        self.EO_dataset = torchvision.datasets.ImageFolder(root=self.EO_dir, transform=transform)
        self.SAR_dataset = torchvision.datasets.ImageFolder(root=self.SAR_dir, transform=transform)
        self.targets = self.EO_dataset.targets
        # print(type(self.EO_dataset[1]), self.EO_dataset[1])
        # print(torch.equal(torch.as_tensor(self.EO_dataset.targets), torch.as_tensor(self.SAR_dataset.targets)))
        # for i in range(len(self.targets)):
        #     if self.EO_dataset.targets[i] != self.SAR_dataset.targets[i]:
        #         print("No match at " + str(i))

        # print("end")
        
    def __len__(self):
        return len(self.EO_dataset)

    def __getitem__(self, idx):
        eo_path, _ = self.EO_dataset.samples[idx]
        # sar_path, _ = self.SAR_dataset.samples[idx]
        eo_image_id = eo_path.split("/")[-1].replace("EO_", "").replace(".png", "")
        # sar_image_id = sar_path.split("/")[-1].replace("SAR_", "").replace(".png", "")
        return self.EO_dataset[idx][0], self.SAR_dataset[idx][0], eo_image_id #, sar_image_id
