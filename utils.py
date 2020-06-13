import os
import pickle
import random
import shutil

import torch
import torch.distributed as dist
from torchvision.transforms._functional_video import resize


def gather_score(x, n=1, divide=True):
    if torch.distributed.is_initialized():
        xn = torch.Tensor([x * n, n]).cuda()
        torch.distributed.all_reduce(xn)
        x, n = xn
        if divide:
            return x / n
        else:
            return x
    else:
        return x


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# From https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/comm.py
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def save_checkpoint(model, optim, epoch, loss, checkpoint_path, global_step, is_best, amp=None, args=None,
                    filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint_path, filename)
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'amp': amp.state_dict() if amp else None,
        'epoch': epoch,
        'best_loss': loss,
        'args': args,
        'global_step': global_step
    }, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint_path, 'model_best.pth'))


def load_checkpoint(model, optim, checkpoint_path, device, amp=None, filename='model_best.pth'):
    filepath = os.path.join(checkpoint_path, filename)
    checkpoint = torch.load(filepath, map_location=device)

    try:
        ret = model.load_state_dict(checkpoint['model'], strict=False)
        if any(('svo_decoder' not in k and 'cvpr_oops_decoder' not in k) for k in ret.missing_keys):
            raise RuntimeError
    except RuntimeError:  # dataparallel vs regular
        d = {}
        for k, v in checkpoint['model'].items():
            d[k.replace('module.', '')] = v
        ret = model.load_state_dict(d, strict=False)
        if any(('svo_decoder' not in k) for k in ret.missing_keys):
            raise RuntimeError
    try:
        optim.load_state_dict(checkpoint['optim'])
    except:
        pass  # happens, e.g. in svo setting
    if amp:
        amp.load_state_dict(checkpoint['amp'])

    epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']

    return epoch, best_loss, checkpoint['global_step']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class RandomResizeVideo:
    def __init__(self, size, interpolation_mode="bilinear"):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)
        self.interpolation = interpolation_mode

    def __call__(self, clip):
        *_, h, w = clip.shape
        size = random.randint(*self.size)
        if (w <= h and w == size) or (h <= w and h == size):
            ow = w
            oh = h
        elif w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return resize(clip, (oh, ow), self.interpolation)

    def __repr__(self):
        return f'RandomResizeVideo(size={self.size}, interpolation={self.interpolation})'


def wordfreq_txt_to_dict(fn):
    with open(fn) as f:
        data = list(f)
    data = [_.strip().split() for _ in data]
    data = {k: int(v) for k, v in data}
    return data
