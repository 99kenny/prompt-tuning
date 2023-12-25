# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from image_prompt_loss import *
import utils
import copy

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, prompt=None, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, frequency_matrix = None, writer=None, optimizers=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()
    optimizer, optimizerE, optimizerG = optimizers
    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if args.lwf == True and prompt is not None:
        metric_logger.add_meter('lwf', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if args.type == 'genp':
        metric_logger.add_meter('reconstruction', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('KLD', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # if args.var:
    #     metric_logger.add_meter('tv_l1', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    #     metric_logger.add_meter('tv_l2', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # if args.l2:
    #     metric_logger.add_meter('l2', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    length = len(data_loader)
    for n_iter, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']
        if 'prompt_idx' in output:
                for batch in output['prompt_idx']: # batch, top_k
                    for idx in batch:
                        frequency_matrix[idx,task_id] += 1 
             
        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']
        if args.type == 'genp':
            alpha_1 = 0.5
            alpha_2 = 0.5
            
            # reconstruction loss
            loss_g1 = (output['synthesized_features'] - output['cls_features']).pow(2).sum(1).mean()
            # KLD
            loss_g2 = -0.5 * (1 + output['var'] - output['mean'].pow(2) - output['var'].exp()).sum(dim=1).mean()
            
            loss = loss + loss_g1 * alpha_1 + loss_g2 * alpha_2
            
            # LwF
            if args.lwf == True and prompt is not None:
                # patched_input = model.patch_embed(input)
                out = prompt(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)['logits']
                if args.train_mask and class_mask is not None:
                    mask = class_mask[task_id]
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    out = torch.index_select(out,1,torch.tensor(mask).to('cuda'))   
                    _logits = torch.index_select(logits,1,torch.tensor(mask).to('cuda'))
                pred = torch.log_softmax(_logits/2, dim=1)
                soft = torch.softmax(out/2, dim=1)
                loss_lwf = -1 * torch.mul(soft, pred).sum() / pred.shape[0] 
    
                loss = loss+ 0.01 * loss_lwf
                metric_logger.meters['lwf'].update(loss_lwf.item(), n=input.shape[0])


            metric_logger.meters['reconstruction'].update(loss_g1.item(), n=input.shape[0])
            metric_logger.meters['KLD'].update(loss_g2.item(), n=input.shape[0])
            
        batch_size, c, h, w = input.shape
        # if args.var:
        #     loss_var_l1, loss_var_l2 = r_prior(model.prompt.prompt[output['prompt_idx']].reshape(batch_size, args.top_k, 3,32,32))
        #     loss = loss + args.alpha_tv_l1 * loss_var_l1 + args.alpha_tv_l2 * loss_var_l2
        #     metric_logger.meters['tv_l1'].update(args.alpha_tv_l1 * loss_var_l1.item(), n=input.shape[0])
        #     metric_logger.meters['tv_l2'].update(args.alpha_tv_l2 * loss_var_l2.item(), n=input.shape[0])
        # if args.l2:
        #     l2 = args.alpha_l2 * r_l2(model.prompt.prompt[output['prompt_idx']].reshape(batch_size, args.top_k, 3, 32,32))
        #     loss = loss + l2
        #     metric_logger.meters['l2'].update(l2.item(), n=input.shape[0])
        # if args.feature:
        #     pass
        #     # loss = loss + args.alpha_f * args.r_feature()
    
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        optimizerG.zero_grad()
        optimizerE.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizerE.step()
        optimizerG.step()
        
        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        if args.type == 'genp':
            metric_logger.meters['reconstruction'].update(loss_g1.item(), n=input.shape[0])
            metric_logger.meters['KLD'].update(loss_g2.item(), n=input.shape[0])

        writer.add_scalar(f'Train/{task_id+1}/Loss', loss.item(), epoch*length + n_iter)
        writer.add_scalar(f'Train/{task_id+1}/Acc1', acc1.item(), epoch*length + n_iter)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, frequency_matrix=None, writer=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()
    i = 0
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            class_matrix = np.zeros((input.shape[0], args.top_k+1))
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            
            class_matrix[:,0] = target.t().clone().detach().cpu()
            class_matrix[:,1:] = output['prompt_idx'].clone().detach().cpu()
            with open(f'{args.output_dir}/{args.exp}/class_mat.csv', "a") as f:
                np.savetxt(f, class_matrix, delimiter=",")
            
            i=i+1
            
            logits = output['logits']
            if 'prompt_idx' in output:
                for batch in output['prompt_idx']: # batch, top_k
                    for idx in batch:
                        frequency_matrix[idx,task_id] += 1 
            
            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask
            
            loss = criterion(logits, target)
            
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0]) 

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, frequency_matrix=None, writer=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, frequency_matrix=frequency_matrix, writer=writer, args=args)
        
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
        writer.add_scalar(f'Task{i+1}/Acc1', stat_matrix[0,i], task_id+1)
        
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)
       
    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
        
        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    
        writer.add_scalar('Avg/Acc1', avg_stat[0], task_id+1)
        writer.add_scalar('Forgetting', forgetting, task_id+1)
    print(result_str)
    
    
    return test_stats



def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, optimizers = None, args = None,):
    writer = SummaryWriter(f'{args.output_dir}/{args.exp}/{args.note}')
    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    frequency_matrix_train = np.zeros((args.size,args.num_tasks))
    frequency_matrix_test = np.zeros((args.size,args.num_tasks))
    
    for task_id in range(args.num_tasks):
       # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))
                
                    with torch.no_grad():
                        if args.distributed:
                            model.module.prompt.prompt.grad.zero_()
                            model.module.prompt.prompt[cur_idx] = model.module.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.prompt.prompt.grad.zero_()
                            model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.prompt.prompt_key.grad.zero_()
                        model.module.prompt.prompt_key[cur_idx] = model.module.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.prompt.prompt_key.grad.zero_()
                        model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            # optimizer = create_optimizer(args, model)
            model_params = []
            for name, p in model_without_ddp.named_parameters():
                if "generator" not in name:
                    model_params.append(p)
            
            # optimizer = create_optimizer(args, model_without_ddp)
            optimizer = torch.optim.AdamW(model_params,lr=args.lr)
            # todo
            optimizerE = torch.optim.AdamW(model_without_ddp.prompt.generator.encoder.parameters(), lr=1e-4)
            optimizerG = torch.optim.AdamW(model_without_ddp.prompt.generator.decoder.parameters(), lr=1e-4)

            optimizers = [optimizer, optimizerE, optimizerG]
        prompt=None
        if task_id > 0:
                if args.lwf:
                    prompt = copy.deepcopy(model)
                    for param in prompt.parameters():
                        param.requires_grad = False
                        
        for epoch in range(args.epochs):            
            train_stats = train_one_epoch(prompt=prompt,model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args, writer=writer, frequency_matrix = frequency_matrix_train, optimizers=optimizers)
            if lr_scheduler:
                lr_scheduler.step(epoch)
        model.prompt.reinit_freq() 
        if args.type == 'imgp' or args.type == 'ptchp' or args.type == 'ptchp_s':
            save_image(make_grid(model.prompt.prompt, nrow=args.size, normalize=True), f'{args.output_dir}/{args.exp}/task{task_id+1}.jpg')
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args, writer=writer, frequency_matrix = frequency_matrix_test)
        # if args.output_dir and utils.is_main_process():
        #     Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
        #     checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
        #     state_dict = {
        #             'model': model_without_ddp.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'epoch': epoch,
        #             'args': args,
        #         }
        #     if args.sched is not None and args.sched != 'constant':
        #         state_dict['lr_scheduler'] = lr_scheduler.state_dict()
        
    np.savetxt(f"{args.output_dir}/{args.exp}/till_task{task_id}_train_freq.csv", frequency_matrix_train, delimiter=",")
    np.savetxt(f"{args.output_dir}/{args.exp}/till_task{task_id}_test_freq.csv", frequency_matrix_test, delimiter=",")
    np.savetxt(f"{args.output_dir}/{args.exp}/till_task{task_id}_acc.csv", np.transpose(acc_matrix), delimiter=",")
        
        #     utils.save_on_master(state_dict, checkpoint_path)

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #     **{f'test_{k}': v for k, v in test_stats.items()},
        #     'epoch': epoch,}

        # if args.output_dir and utils.is_main_process():
        #     with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
