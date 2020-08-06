import random

import torch
from torch.utils.data import Sampler
from torchvision.datasets.video_utils import VideoClips


# Modified from torchvision source code
class UniformClipSamplerMulti(Sampler):
    """
    Sample `num_video_clips_per_video` clips for each video, equally spaced.
    When number of unique clips in the video is fewer than num_video_clips_per_video,
    repeat the clips until `num_video_clips_per_video` clips are collected

    Arguments:
        video_clips (list): list of video clips objects to sample from
        max_clips_per_video (int): number of clips to be sampled per video
    """

    def __init__(self, video_clips, max_clips_per_video):
        for vc in video_clips:
            if not isinstance(vc, VideoClips):
                raise TypeError("Expected video_clips to be a list of instances of VideoClips, "
                                "got {}".format(type(video_clips)))
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self):
        idxs = []
        s = 0
        # select num_clips_per_video for each video, uniformly spaced
        for vc in self.video_clips:
            for i, c in enumerate(vc.clips):
                length = vc.clip_sizes[i]
                if length == 0:
                    # corner case where video decoding fails
                    continue
                sampled = (torch.linspace(s, s + length - 1, steps=min(length, self.max_clips_per_video)).floor().to(
                    torch.int64))
                s += length
                idxs.append(sampled)
        idxs = torch.cat(idxs).tolist()
        return iter(idxs)

    def __len__(self):
        return sum(vc.clip_sizes.clamp(max=self.max_clips_per_video).sum() for vc in self.video_clips)


class RandomClipSamplerMulti(Sampler):
    """
    Samples at most `max_video_clips_per_video` clips for each video randomly

    Arguments:
        video_clips (list): video clips to sample from
        max_clips_per_video (int): maximum number of clips to be sampled per video
    """

    def __init__(self, video_clips, max_clips_per_video):
        for vc in video_clips:
            if not isinstance(vc, VideoClips):
                raise TypeError("Expected video_clips to be a list of instances of VideoClips, "
                                "got {}".format(type(video_clips)))
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self):
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, randomly
        for vc in self.video_clips:
            for i, c in enumerate(vc.clips):
                length = vc.clip_sizes[i]
                size = min(length, self.max_clips_per_video)
                sampled = torch.randperm(length)[:size] + s
                s += length
                idxs.append(sampled)
        idxs = torch.cat(idxs)
        # shuffle all clips randomly
        perm = torch.randperm(len(idxs))
        idxs = idxs[perm].tolist()
        return iter(idxs)

    def __len__(self):  # is it over sampling kinetics because of cropping off the very long videos in oops?
        return sum(vc.clip_sizes.clamp(max=self.max_clips_per_video).sum() for vc in self.video_clips)
