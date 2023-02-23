import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import moviepy.editor as mp
import numpy as np
import pydub
import torch.nn.functional as F


class KineticsDataset(Dataset):
    def __init__(
            self,
            root_dir,
            annotations_path,
            video_transforms=None,
            audio_transforms=None,
            frames_per_clip=16,
            audio_sampling_rate=16000,
            audio_duration=10,
            is_train=False):
        self.root_dir = os.path.join(root_dir, "train" if is_train else "test")
        self.num_frames = frames_per_clip
        self.video_transforms = video_transforms
        self.audio_transforms = audio_transforms
        self.audio_duration = audio_duration
        self.audio_sampling_rate = audio_sampling_rate
        self.n_audio_samples = self.audio_duration * self.audio_sampling_rate
        self.annotations_path = annotations_path
        self.annotations = self._get_annotations()
        self.video_files = self._get_video_files()  # File path and label
        # print(len(self.video_files))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_path, label = self.video_files[idx]
        video = mp.VideoFileClip(video_path)

        # Extract 16 frames evenly spaced throughout the video
        n_frames = self.num_frames
        duration = video.duration
        frame_times = [t * duration / (n_frames + 1) for t in range(1, n_frames + 1)]

        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])

        frames = torch.from_numpy(np.asarray([video.get_frame(t) for t in frame_times], dtype=np.float32))
        # frames = torch.stack([video.get_frame(t) for t in frame_times])
        # print(frames.shape, type(frames))

        # Extract the audio and convert it to mono channel
        audio = pydub.AudioSegment.from_file(video_path).set_channels(1).set_frame_rate(self.audio_sampling_rate)
        audio = torch.FloatTensor(audio.get_array_of_samples())
        if audio.shape[0] >= self.n_audio_samples:
            audio = audio[:self.n_audio_samples]
        else:
            audio = F.pad(input=audio, pad=(0, self.n_audio_samples - audio.shape[0]), mode='constant', value=0)

        # Apply Transforms
        video = frames
        if self.video_transforms is not None:
            video = self.video_transforms(video)
        if self.audio_transforms is not None:
            audio = self.audio_transforms(audio)

        # return video, audio, label
        return video, audio, label

    def _get_annotations(self):
        annotations_map = {}
        with open(self.annotations_path) as f:
            for line in f:
                name, index = line.split()
                annotations_map[name] = index

        return annotations_map

    def _get_video_files(self):
        video_files = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.mp4'):
                        video_path = os.path.join(subdir_path, filename)
                        label = int(self.annotations[subdir])
                        video_files.append((video_path, label))
        return video_files


# KineticsDataset("../datasets/kinetics-400", "../datasets/kinetics-400/annotations.txt", is_train=True)
# KineticsDataset("../datasets/kinetics-400", "../datasets/kinetics-400/annotations.txt", is_train=False)

## Testing Purpose
# import moviepy.editor as mp
# import pydub
# import torch
# import torchvision.transforms as transforms
#
# # Define the input video file path
# video_file_path = "./test.mp4"
#
# # Load the video file
# video = mp.VideoFileClip(video_file_path)
#
# # Extract 16 frames evenly spaced throughout the video
# n_frames = 16
# duration = video.duration
# print("Video duration and fps:", video.duration, video.fps)
# frame_times = [t * duration / (n_frames + 1) for t in range(1, n_frames + 1)]
# frames = [video.get_frame(t) for t in frame_times]
#
# # Extract the audio and convert it to mono channel
# audio = pydub.AudioSegment.from_file(video_file_path).set_channels(1).set_frame_rate(16000)
# print("Frame width:", audio.frame_rate, "Frame rate:", audio.frame_width, "Channel:", audio.channels, "Duration:", audio.duration_seconds)
# audio = torch.FloatTensor(audio.get_array_of_samples())
#
# num_samples = 10*16000
# if audio.shape[0] >= num_samples:
#     audio = audio[:num_samples]
# else:
#     audio = F.pad(input=audio, pad=(0, num_samples-audio.shape[0]), mode='constant', value=0)
#
# # Convert the frames and audio to PyTorch tensors
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
#
# frames = torch.stack([transform(frame) for frame in frames])
#
# # Print the shapes of the tensors
# print(frames.shape)
# print("Shape:", audio.shape, audio)
