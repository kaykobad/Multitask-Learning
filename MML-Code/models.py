import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel, ASTModel


class FusionMethod:
    concat = "CONCAT"
    mlp = "MLP"


def patch_embedding(frames, patch_size):



def patchify(frames, patch_size):
    b, f, c, h, w = frames
    n_patches = f * h * w // (patch_size ** 2)
    patch_dim = c * patch_size * patch_size
    images = frames.view(b, c, h, w*f)
    n, c, h, w = images.shape

    patches = torch.zeros(n, n_patches, patch_dim)

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


class FusionLayer(nn.Module):
    def __int__(self, method):
        self.method = method

    def forward(self, x_video, x_audio):
        out = torch.cat((x_video, x_audio), 1)
        print("Video, audio, output shape from fusion layer: ", x_video.shape, x_audio.shape, out.shape)


class CommonSpaceProjectionLayer(nn.Module):
    def __int__(self):
        pass

    def forward(self, x):
        pass


class ClassificationHead(nn.Module):
    def __int__(self, input_dim, num_classes):
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class ModalitySpecificTransformer(nn.Module):
    def __int__(self, num_classes):
        self.vit_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.ast_encoder = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.fusion = FusionLayer(FusionMethod.concat)

        # TODO: Implement Later
        self.projection = CommonSpaceProjectionLayer()
        self.classification_head = ClassificationHead(768*2, num_classes)

    def forward(self, x_video, x_audio):
        out_video = self.vit_encoder(x_video)
        out_audio = self.ast_encoder(x_audio)
        out = self.fusion(out_video['pooler_output'], out_audio['pooler_output'])
        out = self.classification_head(out)

        return {
            "audio_feature": out_audio,
            "video_feature": out_video,
            "output": out,
        }
