# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import math
from collections import defaultdict, deque
import datetime
import random

import torch
import torch.distributed as dist
from torchvision import transforms
from tqdm import tqdm
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
   
def initialize_prompt_from_training_img(dataset, pool_size, length, args):
    from datasets import build_transform, get_dataset
    # build transform
    transform = transforms.Compose([
            transforms.ToTensor(),
    ]) 
    dataset, _ = get_dataset(dataset, transform, transform, args)
    
    # cls prompt - pool_size * length 만큼 필요 [pool_size, length, embed_dim]
    # selected_idx = [0,6,11,13,20,26,43,47,49,62]
    # idx = [2,3,32,42,55,57,63,141,150,240]
    idx = [29,4,6,9,3,27,0,7,8,1]
    prompt_pool = []
    for i in idx:
        prompt_pool.append(dataset.__getitem__(i)[0].unsqueeze(0))
    # for idx,i in enumerate(dataset):
    #     if i[1] in selected_idx:
    #         print(idx)
    #         selected_idx.remove(i[1])
    #         prompt_pool.append(i[0].unsqueeze(0))
    prompts = torch.cat(prompt_pool, dim=0)
    
    return prompts
 
def initialize_prompt_from_training_img_cls(dataset, pool_size, length, original_model, args):
    from datasets import build_transform, get_dataset
    # build transform
    transform = build_transform(False, args)   
    dataset, _ = get_dataset(dataset, transform, transform, args)

    # cls prompt - pool_size * length 만큼 필요 [pool_size, length, embed_dim]
    selected_idx = random.sample(range(0,len(dataset)), pool_size*length)
    prompts = []
    print('start loading images for prompt initialization')
    for i in tqdm(selected_idx):
        prompts.append(original_model(dataset.__getitem__(i)[0].unsqueeze(0))['pre_logits'])
    prompts = torch.cat(prompts, dim=0)
    prompts = prompts.reshape(pool_size, length, -1)
    print("prompts shape : ",prompts.shape)
    return prompts
    
# def initialize_prompt_from_training_img(dataset, pool_size, length, args):
#     from datasets import build_transform, get_dataset
#     # build transform
#     transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])
#     print(dataset)
#     dataset, _ = get_dataset(dataset, transform, transform, args)
    
#     # cls prompt - pool_size * length 만큼 필요 [pool_size, length, embed_dim]
#     selected_idx = random.sample(range(0,len(dataset)), pool_size)
#     prompts = []
#     print('start loading images for prompt initialization')
#     c,h,w = dataset.__getitem__(0)[0].shape
#     for i in tqdm(selected_idx):
#         prompts.append(dataset.__getitem__(i)[0].unsqueeze(0))
#         print(dataset.__getitem__(i)[1])
#     prompts = torch.cat(prompts, dim=0)
#     prompts = prompts.reshape(pool_size,c,h,w)
#     print("prompts shape : ", prompts.shape)
#     return prompts

def l2_normalize(x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm   

def compute_similarity(prompt_embed, x_embed, frequency, diversify):
    prompt_norm = l2_normalize(prompt_embed, dim=1)
    x_embed_norm = l2_normalize(x_embed, dim=1)
    
    similarity = torch.matmul(x_embed_norm, prompt_norm.t())
    
    if diversify:
        handicap = (1 / frequency)
        freq_min = handicap.min()
        freq_max = handicap.max()
        handicap = (handicap - freq_min) / (freq_max - freq_min) + 1 
        similarity = similarity * handicap.to(similarity.device)
    
    return similarity, prompt_norm, x_embed_norm

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()
