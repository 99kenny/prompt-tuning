# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image, make_grid
from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import *
import models
import utils

import warnings

from prompt.cls_prompt import ClsPrompt
from prompt.image_prompt import ImagePrompt
from prompt.generator_prompt import GeneratorPrompt
from prompt.patch_embed_prompt import PatchEmbedPrompt
from prompt.patch_embed_prompt_single import PatchEmbedPromptSingle

warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask = build_continual_dataloader(args)
    
    # initialize original model
    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    
    # set dataset name
    if 'Split-' in args.dataset: 
        dataset_name = args.dataset[6:]
    else:
        dataset_name = args.dataset 
        
    # initialize prompt
    prompt = None
    if args.type == 'l2p':
        pass
    # initialize prompt with image cls - cls_prompt
    elif args.type == 'lwip':
        initial_prompt = utils.initialize_prompt_from_training_img_cls(dataset_name, args.size, args.length, original_model, args)
        prompt = ClsPrompt(embed_dim=args.embed_dim, embedding_key=args.embedding_key, prompt_key=args.prompt_key, pool_size=args.size, 
                           top_k=args.top_k, batchwise_prompt=args.batchwise_prompt, prompt_key_init=args.prompt_key_init, initial_prompt=initial_prompt)
    elif args.type == 'imgp':
        initial_prompt = utils.initialize_prompt_from_training_img(dataset_name, args.size, args.length, args)
        os.makedirs(f'{args.output_dir}/{args.exp}', exist_ok=True)
        save_image(make_grid(initial_prompt, nrow=args.size), f'{args.output_dir}/{args.exp}/initial.jpg')
        prompt = ImagePrompt(length=args.length, embed_dim=args.embed_dim, embedding_key=args.embedding_key, prompt_pool=args.prompt_pool,
                             prompt_key=args.prompt_key, pool_size=args.size, top_k=args.top_k, batchwise_prompt=args.batchwise_prompt, prompt_key_init=args.prompt_key_init, initial_prompt=initial_prompt)
    elif args.type == 'genp':
        initial_prompt = utils.initialize_prompt_from_training_img(dataset_name, args.size, args.length, args)
        os.makedirs(f'{args.output_dir}/{args.exp}', exist_ok=True)
        save_image(make_grid(initial_prompt, nrow=args.size), f'{args.output_dir}/{args.exp}/initial.jpg')
        prompt = GeneratorPrompt(length=args.length, embed_dim=args.embed_dim, embedding_key=args.embedding_key, pool_size=args.size, top_k=args.top_k, img_size=args.input_size, 
                                 batchwise_prompt=args.batchwise_prompt, prompt_key_init=args.prompt_key_init, input_dim=args.embed_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, )
    elif args.type == 'ptchp':
        initial_prompt = utils.initialize_prompt_from_training_img(dataset_name, args.size, args.length, args)
        os.makedirs(f'{args.output_dir}/{args.exp}', exist_ok=True)
        save_image(make_grid(initial_prompt, nrow=args.size), f'{args.output_dir}/{args.exp}/initial.jpg')
        prompt = PatchEmbedPrompt(img_size=args.input_size ,length=args.length, embed_dim=args.embed_dim, embedding_key=args.embedding_key, prompt_key=args.prompt_key, pool_size=args.size, top_k=args.top_k, batchwise_prompt=args.batchwise_prompt, prompt_key_init=args.prompt_key_init, initial_prompt=initial_prompt) 
    elif args.type == 'ptchp_s':
        initial_prompt = utils.initialize_prompt_from_training_img(dataset_name, args.size, args.length, args)
        os.makedirs(f'{args.output_dir}/{args.exp}', exist_ok=True)
        save_image(make_grid(initial_prompt, nrow=args.size), f'{args.output_dir}/{args.exp}/initial.jpg')
        prompt = PatchEmbedPromptSingle(img_size=args.input_size ,length=args.length, embed_dim=args.embed_dim, embedding_key=args.embedding_key, prompt_key=args.prompt_key, pool_size=args.size, top_k=args.top_k, batchwise_prompt=args.batchwise_prompt, prompt_key_init=args.prompt_key_init, initial_prompt=initial_prompt) 

    print(prompt)
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        prompt = prompt,
    )
    original_model.to(device)
    model.to(device)  

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0
    
    
    model_params = []
    for name, p in model_without_ddp.named_parameters():
        if "generator" not in name:
            model_params.append(p)
    
    # optimizer = create_optimizer(args, model_without_ddp)
    optimizer = torch.optim.AdamW(model_params,lr=args.lr)
    # todo
    optimizerE = torch.optim.AdamW(model_without_ddp.prompt.generator.encoder.parameters(), lr=1e-3)
    optimizerG = torch.optim.AdamW(model_without_ddp.prompt.generator.decoder.parameters(), lr=1e-3)
    
    optimizers = [optimizer, optimizerE, optimizerG]
        
    named_params = dict(model.named_parameters())
    
    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, optimizers, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('L2P training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 L2P configs')
    elif config == 'five_datasets_l2p':
        from configs.five_datasets_l2p import get_args_parser
        config_parser = subparser.add_parser('five_datasets_l2p', help='5-Datasets L2P configs')
    elif config == 'cifar100_lwip':
        from configs.cifar100_lwip import get_args_parser
        config_parser = subparser.add_parser('cifar100_lwip', help='CIFAR100 LWIP configs')
    elif config == 'cifar100_imgp':
        from configs.cifar100_imgp import get_args_parser
        config_parser = subparser.add_parser('cifar100_imgp', help='CIFAR100 IMGP configs')
    elif config == 'cifar100_genp':
        from configs.cifar100_genp import get_args_parser
        config_parser = subparser.add_parser('cifar100_genp', help='CIFAR100 GENP configs')
    elif config == 'cifar100_ptchp':
        from configs.cifar100_ptchp import get_args_parser
        config_parser = subparser.add_parser('cifar100_ptchp', help='CIFAR100 PTCHP configs')
    elif config == 'cifar100_ptchp_s':
        from configs.cifar100_ptchp_s import get_args_parser
        config_parser = subparser.add_parser('cifar100_ptchp_s', help='CIFAR100 PTCHP_s configs')
    else:
        raise NotImplementedError
    
    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)