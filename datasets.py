import bisect
import itertools
import json
import os
import random
from pathlib import Path
from statistics import median
from types import MethodType

import av
import torch
import torchvision.transforms._transforms_video as transforms_video
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset

import utils

transforms_video.RandomResizeVideo = utils.RandomResizeVideo


class VideoDataset(VisionDataset):

    def __init__(self, root, train, frames_per_clip=16, step_between_clips=1, frame_rate=16, transform=None,
                 extensions=('mp4',), label_fn=lambda x, *_: x, local_rank=-1, get_label_only=False):
        train_or_val = 'train' if train else 'val'
        root = os.path.join(root, train_or_val)
        self.root = root

        super().__init__(root)

        self.transform = transform
        # Function that takes in __getitem__ idx and returns auxiliary label information in the form of a tensor
        self.label_fn = MethodType(label_fn, self)
        self.get_label_only = get_label_only

        clips_fn = os.path.join(root, f'clips_{train_or_val}_{frames_per_clip}_{step_between_clips}_{frame_rate}.pt')

        try:
            self.video_clips = torch.load(clips_fn)
        except FileNotFoundError:
            video_list = list(
                map(str, itertools.chain.from_iterable(Path(root).rglob(f'*.{ext}') for ext in extensions)))
            random.shuffle(video_list)
            if local_rank <= 0:
                print('Generating video clips file: ' + clips_fn)
            self.video_clips = VideoClips(
                video_list,
                frames_per_clip,
                step_between_clips,
                frame_rate,
                num_workers=32
            )
            torch.save(self.video_clips, clips_fn)

        clip_lengths = torch.as_tensor([len(v) for v in self.video_clips.clips])
        self.video_clips.clip_sizes = clip_lengths

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        if self.get_label_only:
            return torch.Tensor([0]), torch.Tensor([0]), self.label_fn(idx)

        try:
            video, audio, info, video_idx = self.video_clips.get_clip(idx)  # Takes in index w.r.t orig clip sizes
        except IndexError as e:
            # Off by one bug in VideoClips object
            vi, ci = self.video_clips.get_clip_location(idx)
            self.video_clips.resampling_idxs[vi][ci][-1] -= 1
            video, audio, info, video_idx = self.video_clips.get_clip(idx)

        if self.transform is not None:
            video = self.transform(video)

        return video, torch.Tensor([0]), self.label_fn(idx)

    def update_subset(self, paths, path_transform=None):
        paths = set(paths)
        for i, path in enumerate(self.video_clips.video_paths):
            if path_transform:
                path = path_transform(path)
            if path not in paths:
                self.video_clips.clip_sizes[i] = 0
        self.video_clips.cumulative_sizes = self.video_clips.clip_sizes.cumsum(0).tolist()

    def use_partial_data(self, fraction):
        self.update_subset(self.video_clips.video_paths[:round(fraction * len(self.video_clips.video_paths))])


''' 
This dataset returns multiple subsequent clips from the same video with configurable length and spacing
'''


class MultiVideoDataset(VideoDataset):
    def __init__(self, min_num_clips, max_num_clips, num_clips_between_clips, max_clips_per_video=None,
                 max_total_clips=None, entire_video_as_clip=False, return_batch_details=False, svo=False, svo_path=None,
                 *args, **kwargs):
        # Don't allow picking last num_vids-1 videos as an item, since they will be sampled in the __getitem__
        # As subsequent parts of the multi-video data
        super().__init__(*args, **kwargs)
        self.entire_video_as_clip = entire_video_as_clip
        self.return_batch_details = return_batch_details
        self.min_num_clips = min_num_clips
        self.max_num_clips = max_num_clips
        self.num_clips_between_clips = num_clips_between_clips

        # Patch VideoClips
        self.video_clips.orig_cumulative_sizes = self.video_clips.cumulative_sizes
        self.video_clips.orig_clip_sizes = torch.as_tensor([len(v) for v in self.video_clips.clips])
        self.video_clips.get_new_clip_location = self.video_clips.get_clip_location
        self.video_clips.get_clip_location = MethodType(get_orig_clip_location, self.video_clips)
        self.video_clips.new_num_clips = self.video_clips.num_clips
        self.video_clips.num_clips = MethodType(num_orig_clips, self.video_clips)

        # Video clips are sampled using the cumulative sizes property which is a cumsum on lengths of each clip
        # We can trick the system into thinking a video is shorter than it is
        # by shortening lengths and recomputing cumsum
        # Since we define a path by its first clip, we do not want to be able to select the last K-1 clips
        # for a path of length K, since there is no full path available starting from those clips.
        # Here, we compute K (self.clip_span) and remove the last K-1 clips from being available for each video

        # Can't start an item in the ${buffer size} last clips of a video
        if entire_video_as_clip:
            clip_lengths = [1 for v in self.video_clips.clips]
        else:
            self.min_clip_span = 1 + (min_num_clips - 1) * num_clips_between_clips
            self.max_clip_span = 1 + (max_num_clips - 1) * num_clips_between_clips
            clip_lengths = [max(0, len(v) - self.min_clip_span + 1) for v in self.video_clips.clips]
            # This is taken care of in sampler, but commented out just in case I'm missing something
            # if max_clips_per_video:
            #     clip_lengths = [min(max_clips_per_video, l) for l in clip_lengths]
        clip_lengths = torch.as_tensor(clip_lengths)
        cumsum = clip_lengths.cumsum(0)
        if max_total_clips is not None:
            clip_lengths[cumsum > max_total_clips] = 0
            cumsum = cumsum[cumsum <= max_total_clips]
        self.video_clips.cumulative_sizes = cumsum.tolist()
        self.video_clips.clip_sizes = clip_lengths

        self.svo = svo
        if svo:
            self.svo_eval = False
            with open(svo_path) as f:
                self.svo_data = json.load(f)
                self.svo_data = {k.encode("cp1252").decode("utf-8"): v for k, v in self.svo_data.items()}
            with open(os.path.join(os.path.dirname(svo_path), 'vocab.json')) as f:
                self.svo_vocab = json.load(f)
            with open(os.path.join(os.path.dirname(svo_path), 'corrections.json')) as f:
                self.svo_corrs = json.load(f)

    def __getitem__(self, idx):
        idx = self.new_idx_to_orig(idx)
        data = []
        vidx, _ = self.video_clips.get_clip_location(idx)
        if self.entire_video_as_clip:
            span = min(self.max_num_clips, max(self.min_num_clips, self.video_clips.orig_clip_sizes[vidx]))
        else:
            span = random.randint(self.min_clip_span, self.max_clip_span)
        vid_ts = []
        if self.return_batch_details:
            video_path = self.video_clips.video_paths[self.video_clips.get_clip_location(idx)[0]]
            t_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
        for _idx in range(idx, idx + span, self.num_clips_between_clips):
            _vidx, _ = self.video_clips.get_clip_location(_idx)
            if _vidx != vidx:
                break
            if self.return_batch_details:
                video_idx, clip_idx = self.video_clips.get_clip_location(_idx)
                clip_pts = self.video_clips.clips[video_idx][clip_idx]
                t_start = clip_pts[0].item() * t_unit
                t_end = clip_pts[-1].item() * t_unit
                vid_ts.append((t_start + t_end) / 2)
            data.append(VideoDataset.__getitem__(self, _idx))
        videos, audios, labels = zip(*data)
        # return torch.stack(videos), torch.stack(audios)
        # if self.return_batch_details:
        #     return videos, labels, len(videos), vid_ts  # We don't use audio
        # else:
        #     return videos, labels, len(videos)
        d = {
            'videos': videos,
            'labels': labels,
            'len': len(videos)
        }
        if self.return_batch_details:
            d['ts'] = vid_ts
        if self.svo:
            svo_dicts = self.svo_data[self.video_clips.video_paths[vidx].rsplit(os.path.sep, 1)[1].rsplit('.', 1)[0]]
            goal_labels = []
            wentwrong_labels = []
            if self.svo_eval:
                for svo_dict in svo_dicts:
                    for goal_svo in svo_dict['kgoalsvos']:
                        goal_labels.append(
                            [self.svo_vocab.index(w if w in self.svo_vocab else self.svo_corrs[w]) for w in goal_svo])
                    for wentwrong_svo in svo_dict['kwentwrongsvos']:
                        wentwrong_labels.append(
                            [self.svo_vocab.index(w if w in self.svo_vocab else self.svo_corrs[w]) for w
                                in
                                wentwrong_svo])
                max_len = max(len(goal_labels), len(wentwrong_labels))
                padding = [[-1, -1, -1] for _ in range(max_len)]
                goal_labels = (goal_labels + padding)[:max_len] or [[-1, -1, -1]]
                wentwrong_labels = (wentwrong_labels + padding)[:max_len] or [[-1, -1, -1]]
            else:
                svo_dict = random.choice(svo_dicts)
                try:
                    goal_svo = random.choice(svo_dict['kgoalsvos'])
                    goal_labels = [self.svo_vocab.index(w if w in self.svo_vocab else self.svo_corrs[w]) for w in
                        goal_svo]
                except:
                    goal_labels = [-1, -1, -1]
                try:
                    wentwrong_svo = random.choice(svo_dict['kwentwrongsvos'])
                    wentwrong_labels = [self.svo_vocab.index(w if w in self.svo_vocab else self.svo_corrs[w]) for w in
                        wentwrong_svo]
                except:
                    wentwrong_labels = [-1, -1, -1]
            d['svo_labels'] = torch.tensor([(goal_labels if l == 0 else wentwrong_labels) for l in labels])
        return d

    def new_idx_to_orig(self, idx):
        video_idx, clip_idx = self.video_clips.get_new_clip_location(idx)
        orig_idx = clip_idx
        if video_idx > 0:
            orig_idx += self.video_clips.orig_cumulative_sizes[video_idx - 1]
        return orig_idx

    def __len__(self):
        return self.video_clips.new_num_clips()


class KineticsAndOops(VisionDataset):
    def __init__(self, kinetics_root, oops_root, min_num_clips, max_num_clips, train, kinetics_label_fn=None,
                 oops_label_fn=None, use_oops_gt=False, use_oops_only=False, use_kinetics_only=False, data_fraction=1,
                 return_batch_details=False, use_pseudo_oops_gt=False, svo=False, svo_path=None,
                 cache_pre_svo_reprs=False, **kwargs):
        super().__init__(kinetics_root + ' and ' + oops_root)

        self.reprs_cached = False
        self.return_batch_details = return_batch_details
        self.cache_pre_svo_reprs = cache_pre_svo_reprs
        self.svo = svo

        self.oops = MultiVideoDataset(min_num_clips, max_num_clips, root=oops_root, label_fn=oops_label_fn, train=train,
                                      return_batch_details=return_batch_details, svo=svo,
                                      svo_path=os.path.join(svo_path,
                                                            f'{"train" if train else "val"}.json') if svo else None,
                                      **kwargs)
        with open(os.path.join(oops_root, 'all_mturk_data.json')) as f:
            self.oops.oops_data = json.load(f)
        if use_oops_gt:
            if use_pseudo_oops_gt and train:
                with open(os.path.join(oops_root, 'pseudo_train_mturk_data.json')) as f:
                    for k, v in json.load(f).items():
                        self.oops.oops_data[k] = v
            self.oops.update_subset([k for k, v in self.oops.oops_data.items() if 0.01 <= median(v['rel_t']) <= 0.99
                                                                                  and 3 < v['len'] < 30],
                                    lambda fn: os.path.splitext(os.path.basename(fn))[0])

        if 0 < data_fraction < 1:
            self.oops.use_partial_data(data_fraction)
        self.oops_len = len(self.oops)
        if use_kinetics_only:
            self.oops_len = 0
            self.kinetics = MultiVideoDataset(min_num_clips, max_num_clips, root=kinetics_root,
                                              label_fn=kinetics_label_fn, train=train,
                                              return_batch_details=return_batch_details,
                                              **kwargs)

            self.kinetics_len = len(self.kinetics)

            self.video_clips = [self.kinetics.video_clips]  # Order is important!
            del self.oops
        elif use_oops_only:
            self.kinetics_len = 0
            self.video_clips = [self.oops.video_clips]
        else:
            self.kinetics = MultiVideoDataset(min_num_clips, max_num_clips, root=kinetics_root,
                                              label_fn=kinetics_label_fn,
                                              max_total_clips=self.oops_len, train=train,
                                              return_batch_details=return_batch_details,
                                              **kwargs)

            self.kinetics_len = len(self.kinetics)

            self.video_clips = [self.kinetics.video_clips, self.oops.video_clips]  # Order is important!

    def collate_fn(self, batch):
        videos = []
        labels = []
        svo_labels = []
        lens = []
        video_types = []
        idxs = []
        video_ts = []
        for d in batch:
            videos += d['videos']
            labels += d['labels']
            if self.svo:
                svo_labels += d['svo_labels']
            lens.append(d['len'])
            video_types.append(d['video_type'])
            idxs.append(d['idx'])
            if self.return_batch_details:
                video_ts += d['ts']
        videos = torch.stack(videos)
        labels = torch.stack(labels)
        if self.svo:
            try:
                svo_labels = torch.stack(svo_labels).permute(1, 0, 2)
            except:
                try:
                    svo_labels = pad_sequence(svo_labels, batch_first=False, padding_value=-1)
                except:
                    pad = torch.full_like([_ for _ in svo_labels if len(_)][0], -1)
                    for i, l in enumerate(svo_labels):
                        if not len(l):
                            svo_labels[i] = pad
                    try:
                        svo_labels = pad_sequence(svo_labels, batch_first=False, padding_value=-1)
                    except:
                        import ipdb
                        ipdb.set_trace()
                # print(svo_labels.shape)
        lens = torch.LongTensor(lens)
        video_types = torch.LongTensor(video_types)
        idxs = torch.LongTensor(idxs)
        video_ts = torch.Tensor(video_ts)
        if self.return_batch_details:
            if self.svo:
                details = (video_ts,)
            else:
                details = (idxs, video_ts)
        else:
            details = ()
        if self.svo:
            return (videos, labels, svo_labels, lens, idxs, *details)
        else:
            return (videos, labels, lens, video_types, *details)

    def __len__(self):
        return self.kinetics_len + self.oops_len

    def get_filename_by_idx(self, idx):
        if idx < self.kinetics_len:
            dataset = self.kinetics
        else:
            dataset = self.oops
            idx -= self.kinetics_len
        idx = dataset.new_idx_to_orig(idx)
        vidx, _ = dataset.video_clips.get_clip_location(idx)
        return dataset.video_clips.video_paths[vidx]

    def __getitem__(self, idx):
        if self.cache_pre_svo_reprs and self.reprs_cached:
            self.oops.get_label_only = True
            self.kinetics.get_label_only = True
        if idx < self.kinetics_len:
            d = self.kinetics[idx]
            d['video_type'] = 0
        else:
            d = self.oops[idx - self.kinetics_len]
            d['video_type'] = 1
        d['idx'] = idx
        if self.cache_pre_svo_reprs and self.reprs_cached:
            d['videos'] = self.reprs[idx]
            d['len'] = len(self.reprs[idx])
        return d


def oops_idx_to_label(self, idx):
    video_idx, clip_idx = self.video_clips.get_clip_location(idx)
    video_path = self.video_clips.video_paths[video_idx]
    clip_pts = self.video_clips.clips[video_idx][clip_idx]
    t_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
    t_start = clip_pts[0].item() * t_unit
    t_end = clip_pts[-1].item() * t_unit
    t_fail = median(self.oops_data[os.path.splitext(os.path.basename(video_path))[0]]['t'])
    label = 0
    if t_start <= t_fail <= t_end:
        label = 1
    elif t_start > t_fail:
        label = 2
    return torch.tensor(label)


def train_transform(s):
    return transforms.Compose([
        transforms_video.ToTensorVideo(),
        transforms_video.RandomHorizontalFlipVideo(),
        transforms_video.RandomResizeVideo((s, round(s * 1.5))),
        transforms_video.NormalizeVideo(mean=[0.43216, 0.394666, 0.37645],
                                        std=[0.22803, 0.22145, 0.216989]),
        transforms_video.RandomCropVideo(s)
    ])


def val_transform(s):
    return transforms.Compose([
        transforms_video.ToTensorVideo(),
        transforms_video.RandomResizeVideo(s),
        transforms_video.NormalizeVideo(mean=[0.43216, 0.394666, 0.37645],
                                        std=[0.22803, 0.22145, 0.216989]),
        transforms_video.CenterCropVideo(s)
    ])


'''
VideoClips modifications
'''


def get_orig_clip_location(self, idx):
    """
    Converts a flattened representation of the indices into a video_idx, clip_idx
    representation.
    """
    video_idx = bisect.bisect_right(self.orig_cumulative_sizes, idx)
    if video_idx == 0:
        clip_idx = idx
    else:
        clip_idx = idx - self.orig_cumulative_sizes[video_idx - 1]
    return video_idx, clip_idx


def num_orig_clips(self):
    return self.orig_cumulative_sizes[-1]

#
# class SVOTestDataset(Dataset):
#     def __init__(self, args):
#         path = args.svo_results_path
#         svo_path = os.path.join(args.svo_path, 'val.json')
#         results_fns = sorted(glob(os.path.join(path, 'results*.json')))
#         retrievals_fns = sorted(glob(os.path.join(path, 'retrievals*.pt')))
#         results_data = []
#         retrievals_data = defaultdict(list)
#         for fn in results_fns:
#             with open(fn) as f:
#                 results_data.extend(json.load(f))
#         for fn in retrievals_fns:
#             for k, v in torch.load(fn).items():
#                 try:
#                     retrievals_data[k].extend(v)
#                 except TypeError:
#                     retrievals_data[k].append(v)
#         for fn, v in retrievals_data.items():
#             try:
#                 retrievals_data[fn] = torch.stack(v)
#             except:
#                 if type(v[0]) is list:
#                     retrievals_data[fn] = list(itertools.chain.from_iterable(v))
#                 else:
#                     pass
#         self.data = retrievals_data
#         self.reprs = self.data['cnn_reprs'] if args.svo_cnn_only else self.data['xfmr_reprs']
#         self.ac_reprs = self.data['autocorrect_reprs'] if args.svo_cnn_only else self.data[
#             'autocorrect_xfmr_reprs']
#         self.metadata = results_data
#         self.cumlens = torch.tensor([len(d['gts']) for d in self.metadata]).cumsum(0).tolist()
#         with open(svo_path) as f:
#             self.svo_data = json.load(f)
#             self.svo_data = {k.encode("cp1252").decode("utf-8"): v for k, v in self.svo_data.items()}
#         with open(os.path.join(os.path.dirname(svo_path), 'vocab.json')) as f:
#             self.svo_vocab = json.load(f)
#         with open(os.path.join(os.path.dirname(svo_path), 'corrections.json')) as f:
#             self.svo_corrs = json.load(f)
#
#     def __getitem__(self, i):
#         metadata = self.metadata[i]
#         start_i = self.cumlens[i - 1] if i > 0 else 0
#         end_i = self.cumlens[i]
#         reprs = self.reprs[start_i: end_i]
#         ac_reprs = self.ac_reprs[start_i: end_i]
#         gts = metadata['gts']
#         ts = metadata['ts']
#         svo_dicts = self.svo_data[metadata['fn'].rsplit(os.path.sep, 1)[1].rsplit('.', 1)[0]]
#         goal_labels = []
#         wentwrong_labels = []
#         for svo_dict in svo_dicts:
#             for goal_svo in svo_dict['kgoalsvos']:
#                 goal_labels.append(
#                     [self.svo_vocab.index(w if w in self.svo_vocab else self.svo_corrs[w]) for w in goal_svo])
#             for wentwrong_svo in svo_dict['kwentwrongsvos']:
#                 wentwrong_labels.append(
#                     [self.svo_vocab.index(w if w in self.svo_vocab else self.svo_corrs[w]) for w
#                         in
#                         wentwrong_svo])
#         max_len = max(len(goal_labels), len(wentwrong_labels))
#         padding = [[-1, -1, -1] for _ in range(max_len)]
#         goal_labels = (goal_labels + padding)[:max_len]
#         wentwrong_labels = (wentwrong_labels + padding)[:max_len]
#         svo_labels = torch.tensor([(goal_labels if l == 0 else wentwrong_labels) for l in gts])
#         svo_ac_labels = torch.tensor([goal_labels for l in gts])
#         x = 0
#         return {
#             'reprs': reprs,
#             'ac_reprs': ac_reprs,
#             'gts': gts,
#             'ts': ts,
#             'svo_labels': svo_labels,
#             'svo_ac_labels': svo_ac_labels,
#             'idx': i
#         }
#
#     def __len__(self):
#         return len(self.metadata)
#
#     def collate_fn(self, batch):
#         reprs = []
#         ac_reprs = []
#         gts = []
#         ts = []
#         svo_labels = []
#         svo_ac_labels = []
#         idxs = []
#         lens = []
#         for d in batch:
#             lens.append(len(d['reprs']))
#             reprs += d['reprs']
#             ac_reprs += d['ac_reprs']
#             gts += d['gts']
#             ts += d['ts']
#             svo_labels += d['svo_labels']
#             svo_ac_labels += d['svo_ac_labels']
#             idxs.append(d['idx'])
#         try:
#             svo_labels = pad_sequence(svo_labels, batch_first=False, padding_value=-1)
#             svo_ac_labels = pad_sequence(svo_ac_labels, batch_first=False, padding_value=-1)
#         except:
#             pad = torch.full_like([_ for _ in svo_labels if len(_)][0], -1)
#             for i, l in enumerate(svo_labels):
#                 if not len(l):
#                     svo_labels[i] = pad
#             pad = torch.full_like([_ for _ in svo_ac_labels if len(_)][0], -1)
#             for i, l in enumerate(svo_ac_labels):
#                 if not len(l):
#                     svo_ac_labels[i] = pad
#             svo_labels = pad_sequence(svo_labels, batch_first=False, padding_value=-1)
#             svo_ac_labels = pad_sequence(svo_ac_labels, batch_first=False, padding_value=-1)
#         return dict(
#             reprs=torch.stack(reprs),
#             ac_reprs=torch.stack(ac_reprs),
#             gts=torch.tensor(gts),
#             ts=torch.tensor(ts),
#             svo_labels=svo_labels,
#             svo_ac_labels=svo_ac_labels,
#             lens=torch.tensor(lens),
#             idxs=torch.tensor(idxs)
#         )
