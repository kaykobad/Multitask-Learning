import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel, ASTModel, ASTFeatureExtractor, AutoProcessor
from einops import rearrange, repeat
from torchaudio.transforms import Spectrogram
from datetime import datetime


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class FusionMethod:
    concat = "CONCAT"
    mlp = "MLP"


class PatchEmbed(nn.Module):
    """Images to Patch Embedding.

    Args:
        batch_dim (tuple): Dimension of the batch.
        patch_size (int): Size of one patch.
        tube_size (int): Size of temporal field of one 3D patch.
        in_channels (int): Channel num of input features. Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        conv_type (str): Type for convolution layer. Defaults to 'Conv2d'.
    """

    def __init__(self,
                 batch_dim,
                 patch_size=16,
                 tube_size=1,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv3d'):
        super(PatchEmbed, self).__init__()
        b, t, c, h, w = batch_dim
        self.num_patches = t * h * w // (patch_size ** 2)

        # Use conv layer to embed
        if conv_type == 'Conv2d':
            self.projection = nn.Conv2d(
                in_channels,
                embed_dims,
                kernel_size=patch_size,
                stride=patch_size)
        elif conv_type == 'Conv3d':
            self.projection = nn.Conv3d(
                in_channels,
                embed_dims,
                kernel_size=(tube_size, patch_size, patch_size),
                stride=(tube_size, patch_size, patch_size))
        else:
            raise TypeError(f'Unsupported conv layer type {conv_type}')

        # add class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        # Add positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dims))

        self.init_weights()

    def init_weights(self):
        # if hasattr(module, 'weight') and module.weight is not None:
        #     kaiming_init_(module.weight, mode='fan_in', nonlinearity='relu')
        # if hasattr(module, 'bias') and module.bias is not None:
        #     constant_init_(module.bias, constant_value=0)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        layer_type = type(self.projection)
        if layer_type == nn.Conv3d:
            x = rearrange(x, 'b t c h w -> b c t h w')
            x = self.projection(x)
            x = rearrange(x, 'b c t h w -> b (t h w) c')
        elif layer_type == nn.Conv2d:
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            x = self.projection(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
        else:
            raise TypeError(f'Unsupported conv layer type {layer_type}')

        # Add class token and pos embedding
        cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
        x = x + self.pos_embed
        x = torch.cat((cls_tokens, x), dim=1)

        # print("Embedding shape:", x.shape)

        return x


class AudioFeatureExtraction(nn.Module):
    """Audio to Spectrogram.

    Args:
        sampling_rate (int): Audio Sampling Rate
    """

    def __init__(self, sampling_rate=1600, return_tensors="pt"):
        super(AudioFeatureExtraction, self).__init__()
        self.audio_feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.audio_sampling_rate = sampling_rate

    def forward(self, x):
        batch_size = x.shape[0]
        # p = self.audio_feature_extractor(x[0].cpu())
        # print(type(p), p)
        x = [self.audio_feature_extractor(x[i].cpu(), sampling_rate=self.audio_sampling_rate, return_tensors="pt")['input_values'] for i in range(batch_size)]

        x = torch.cat(x, 0)
        # print("Spectrogram shape:", x.shape)

        return x.to(device)


class FusionLayer(nn.Module):
    def __int__(self, method):
        super(FusionLayer, self).__init__()
        self.method = method

    def forward(self, x_video, x_audio):
        out = torch.cat((x_video, x_audio), 1)
        print("Video, audio, output shape from fusion layer: ", x_video.shape, x_audio.shape, out.shape)


class CommonSpaceProjectionLayer(nn.Module):
    def __int__(self):
        super(CommonSpaceProjectionLayer, self).__init__()

    def forward(self, x):
        pass


class ClassificationHead(torch.nn.Module):

    def __init__(self, input_dim=768, num_classes=101):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=num_classes)

    def forward(self, x):
        return self.linear(x)


class ModalitySpecificTransformer(torch.nn.Module):

    def __init__(self, num_classes=101, batch_dim=(128, 8, 3, 224, 224), patch_size=16, in_channel=3, d_model=768, audio_info=(16000, 10)):
        super(ModalitySpecificTransformer, self).__init__()
        self.audio_sampling_rate, self.audio_duration = audio_info

        vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.video_embedding = PatchEmbed(batch_dim=batch_dim, patch_size=patch_size, in_channels=in_channel, embed_dims=d_model)
        self.vit_encoder = vit.encoder
        self.layernorm = vit.layernorm
        self.pooler = vit.pooler

        self.audio_feature_extractor = AudioFeatureExtraction(sampling_rate=self.audio_sampling_rate)
        # self.audio_feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        # self.audio_embedding = AudioEmbed()
        self.ast_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        # self.fusion = FusionLayer()

        # TODO: Implement Later
        # self.projection = CommonSpaceProjectionLayer()
        self.classification_head = ClassificationHead(input_dim=2*768, num_classes=num_classes)

    def forward(self, v, a):
        start_time = datetime.now()
        v = self.video_embedding(v)
        diff = datetime.now() - start_time
        print("\nVideo Embedding shape:", v.shape, "Took time(s):", diff.total_seconds())
        start_time = datetime.now()
        v = self.vit_encoder(v)
        # sequence_output = v[0]
        sequence_output = self.layernorm(v[0])
        pooled_output_v = self.pooler(sequence_output)
        diff = datetime.now() - start_time
        print("ViT Encoder output shape:", pooled_output_v.shape, "Took time(s):", diff.total_seconds())
        # print("M1:", a.shape)
        # a = rearrange(a, 'b c n -> n c b')
        # print("M1/2:", a.shape)

        # a = self.audio_embedding(a.cpu(), return_tensors="pt")
        # print("M2:", a.shape)
        # a_out = self.ast_encoder(a)
        # print("M3:", a_out.shape)

        # TODO: Audio
        start_time = datetime.now()
        a = self.audio_feature_extractor(a)
        diff = datetime.now() - start_time
        print("\nSpectrogram shape:", a.shape, "Took time(s):", diff.total_seconds())
        start_time = datetime.now()
        a = self.ast_model(a)
        pooled_output_a = a['pooler_output']
        diff = datetime.now() - start_time
        print("AST Encoder output shape:", pooled_output_a.shape, "Took time(s):", diff.total_seconds())

        start_time = datetime.now()
        pooled_output = torch.cat((pooled_output_v, pooled_output_a), dim=1)
        out = self.classification_head(pooled_output)
        diff = datetime.now() - start_time
        print("Final output shape:", out.shape, "Took time(s):", diff.total_seconds())

        # return {
        #     # "audio_feature": out_audio,
        #     "video_feature": v,
        #     "output": out,
        # }
        return out
