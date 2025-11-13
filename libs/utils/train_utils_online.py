# online(pcml)
import os
import shutil
import time
import pickle

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from libs.datasets import to_segments, to_frame_wise
################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # torch.use_deterministic_algorithms(True, warn_only=True)
        torch.use_deterministic_algorithms(False)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, MaskedConv1D, torch.nn.Embedding,torch.nn.Parameter)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d, torch.nn.InstanceNorm1d, torch.nn.Dropout)

    # loop over all modules / params
    for mn, m in model.named_modules():
        # 只枚举当前模块自身的参数，避免重复处理
        for pn, p in m.named_parameters(recurse=False):
            fpn = '%s.%s' % (mn, pn) if mn else pn  # 完整参数名

            # 先处理PCML与在线分割头，避免后续通用规则再命中
            if ('pcml_layer' in fpn or 'pcml_modules' in fpn or
                'pcml_branch_modules' in fpn or 'online_seg_head' in fpn
                or 'pcml_module' in fpn):
                
                # 针对PCML核心参数的精细分类
                if pn in ('W1',):
                    decay.add(fpn)
                elif pn in ('b1', 'learnable_pcml_lr_weight', 'learnable_token_idx',
                            'pcml_norm_weight', 'pcml_norm_bias'):
                    no_decay.add(fpn)
                # PCML中的LayerNorm后处理（post_norm）无权重衰减
                elif isinstance(m, (LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d, torch.nn.InstanceNorm1d)):
                    if pn.endswith('weight') or pn.endswith('bias'):
                        no_decay.add(fpn)
                # 其它可训练权重遵循白名单进入decay
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)

            elif pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('_gmm'):
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            # else:
            #     print(f"pn:{fpn}")

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        self.decay = decay
        self.device = device

        # 维护影子权重，不进行deepcopy模型本身
        self.shadow = {}
        for k, v in model.state_dict().items():
            s = v.detach().clone()
            if self.device is not None:
                s = s.to(self.device)
            self.shadow[k] = s

    def _update(self, model, update_fn):
        with torch.no_grad():
            msd = model.state_dict()
            for k in self.shadow.keys():
                mv = msd[k]
                if self.device is not None:
                    mv = mv.to(self.device)
                self.shadow[k].copy_(update_fn(self.shadow[k], mv))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def apply_to(self, model):
        # 可选：将EMA权重加载回模型，用于验证或保存
        model.load_state_dict(self.shadow, strict=True)

def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 10, #20,
    use_contrastive = False,
    batch_size = 2,
    max_videos = 15,
    mode: str = 'offline',  # 新增：训练模式（offline/online）
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    num_videos = 0
    start = time.time()

    for iter_idx, video_list in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        # use CSPL, generating prototypes
        if use_contrastive and num_videos < max_videos * batch_size:
            assert max_videos < len(train_loader)
            if num_videos == 0:
                model(video_list, mode='clustering_init')
            
            model(video_list, mode='clustering')
            num_videos += batch_size
            
            if num_videos >= max_videos * batch_size:
                model(video_list, mode='clustering_flush')

        # if not using CSPL, or prototypes generation is complete
        if not use_contrastive or num_videos >= max_videos * batch_size:
            if mode == 'online':
                # 在线分支：前向得到在线logits与掩码
                output = model(video_list, mode='online')
                online_logits = output['online_logits']       # [B, T, C]
                fpn_mask = output['fpn_mask']                 # [B, T] (0/1)

                # 构造帧级GT标签（默认忽略非片段帧）
                B, T, C = online_logits.shape
                gt = torch.full((B, T), -100, dtype=torch.long, device=online_logits.device)  # ignore_index
                for b in range(B):
                    segs = video_list[b]['segments']          # Tensor [N, 2]
                    labels = video_list[b]['labels']          # Tensor [N]
                    # 注意：只处理合法片段范围
                    for j in range(segs.shape[0]):
                        s = int(segs[j, 0].item())
                        e = int(segs[j, 1].item())
                        if s < e:
                            cls = int(labels[j].item())
                            s_clamped = max(0, min(T, s))
                            e_clamped = max(0, min(T, e))
                            gt[b, s_clamped:e_clamped] = cls
                    # 掩码为0的帧也忽略
                    gt[b, fpn_mask[b] == 0] = -100

                # 计算交叉熵损失（忽略-100）
                ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = ce(online_logits.reshape(B * T, C), gt.reshape(B * T))
                losses = {'final_loss': loss}
            else:
                losses = model(video_list)

            losses['final_loss'].backward()
            # gradient cliping (to stabilize training if necessary)
            if clip_grad_l2norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_grad_l2norm
                )
            optimizer.step()
            scheduler.step()

            if model_ema is not None:
                model_ema.update(model)

            # printing (only check the stats when necessary to avoid extra cost)
            if (iter_idx != 0) and (iter_idx % print_freq) == 0:
                # measure elapsed time (sync all kernels)
                torch.cuda.synchronize()
                batch_time.update((time.time() - start) / print_freq)
                start = time.time()

                # track all losses
                for key, value in losses.items():
                    # init meter if necessary
                    if key not in losses_tracker:
                        losses_tracker[key] = AverageMeter()
                    # update
                    losses_tracker[key].update(value.item())

                # log to tensor board
                lr = scheduler.get_last_lr()[0]
                global_step = curr_epoch * num_iters + iter_idx
                if tb_writer is not None:
                    # learning rate (after stepping)
                    tb_writer.add_scalar(
                        'train/learning_rate',
                        lr,
                        global_step
                    )
                    # all losses
                    tag_dict = {}
                    for key, value in losses_tracker.items():
                        if key != "final_loss":
                            tag_dict[key] = value.val
                    tb_writer.add_scalars(
                        'train/all_losses',
                        tag_dict,
                        global_step
                    )
                    # final loss
                    tb_writer.add_scalar(
                        'train/final_loss',
                        losses_tracker['final_loss'].val,
                        global_step
                    )

                # print to terminal
                block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                    curr_epoch, iter_idx, num_iters
                )
                block2 = 'Time {:.2f} ({:.2f})'.format(
                    batch_time.val, batch_time.avg
                )
                block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                    losses_tracker['final_loss'].val,
                    losses_tracker['final_loss'].avg
                )
                block4 = ''
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        block4  += '\t{:s} {:.2f} ({:.2f})'.format(
                            key, value.val, value.avg
                        )

                print('\t'.join([block1, block2, block3, block4]))
                # if final_losses is not None:
                #     final_losses.append(losses_tracker['final_loss'].val)

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))

def train_one_epoch_gmm(
    train_loader,
    model,
    ckpt_folder
):
    """Training the model for one epoch"""

    # switch to train mode
    model.train()
    print("\n[Train]: GMM Model started")
    start = time.time()

    for iter_idx, video_list in enumerate(train_loader):
        with torch.no_grad():        
            if iter_idx == 0:
                model(video_list, mode='gmm_init')

            model(video_list, mode='gmm')

            if iter_idx == len(train_loader) - 1:
                model(video_list, mode='gmm_fit')
    model.module.gmm_save(ckpt_folder)
    end = time.time()
    print(f"[Train]: GMM Model finished. Time: {((end-start)/60):.2f} minutes")




def train_one_epoch_backup(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 10, #20,
    use_contrastive = False,
    batch_size = 2,
    max_videos = 15,
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    num_videos = 0
    start = time.time()

    for iter_idx, video_list in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        # use CSPL, generating prototypes
        if use_contrastive:
            assert max_videos < len(train_loader)
            if num_videos == 0:
                model(video_list, mode='clustering_init')
            
            if num_videos < max_videos * batch_size:
                model(video_list, mode='clustering')
                num_videos += batch_size
            
            if num_videos == max_videos * batch_size:
                model(video_list, mode='clustering_flush')
                num_videos += batch_size

        # if not using CSPL, or prototypes generation is complete
        if not use_contrastive or num_videos > max_videos * batch_size:
            
            losses = model(video_list)
            losses['final_loss'].backward()
            # gradient cliping (to stabilize training if necessary)
            if clip_grad_l2norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_grad_l2norm
                )
            # step optimizer / scheduler
            optimizer.step()
            scheduler.step()

            if model_ema is not None:
                model_ema.update(model)

            # printing (only check the stats when necessary to avoid extra cost)
            if (iter_idx != 0) and (iter_idx % print_freq) == 0:
                # measure elapsed time (sync all kernels)
                torch.cuda.synchronize()
                batch_time.update((time.time() - start) / print_freq)
                start = time.time()

                # track all losses
                for key, value in losses.items():
                    # init meter if necessary
                    if key not in losses_tracker:
                        losses_tracker[key] = AverageMeter()
                    # update
                    losses_tracker[key].update(value.item())

                # log to tensor board
                lr = scheduler.get_last_lr()[0]
                global_step = curr_epoch * num_iters + iter_idx
                if tb_writer is not None:
                    # learning rate (after stepping)
                    tb_writer.add_scalar(
                        'train/learning_rate',
                        lr,
                        global_step
                    )
                    # all losses
                    tag_dict = {}
                    for key, value in losses_tracker.items():
                        if key != "final_loss":
                            tag_dict[key] = value.val
                    tb_writer.add_scalars(
                        'train/all_losses',
                        tag_dict,
                        global_step
                    )
                    # final loss
                    tb_writer.add_scalar(
                        'train/final_loss',
                        losses_tracker['final_loss'].val,
                        global_step
                    )

                # print to terminal
                block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                    curr_epoch, iter_idx, num_iters
                )
                block2 = 'Time {:.2f} ({:.2f})'.format(
                    batch_time.val, batch_time.avg
                )
                block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                    losses_tracker['final_loss'].val,
                    losses_tracker['final_loss'].avg
                )
                block4 = ''
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        block4  += '\t{:s} {:.2f} ({:.2f})'.format(
                            key, value.val, value.avg
                        )

                print('\t'.join([block1, block2, block3, block4]))
                # if final_losses is not None:
                #     final_losses.append(losses_tracker['final_loss'].val)

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))


def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    results = {}
    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        with torch.no_grad():
            output = model(video_list)

            # print(f"output:{output}")

            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    video_id = output[vid_idx]['video_id']
                    if video_id not in results:
                        results[video_id] = {}

                    preds = to_frame_wise(output[vid_idx]['segments'], output[vid_idx]['labels'],
                                        output[vid_idx]['scores'], video_list[vid_idx]['feats'].size(1), 
                                        fps=video_list[vid_idx]['fps'])
                    # action_labels, time_stamp_labels = generate_time_stamp_labels(preds, -2)
                    action_labels, time_stamp_labels = to_segments(preds)
                    results[video_id]['segments'] = time_stamp_labels
                    results[video_id]['label'] = action_labels
                    results[video_id]['score'] = output[vid_idx]['scores'].numpy()


        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    if evaluator is not None:
        if ext_score_file is not None and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        _, mAP, _ = evaluator.evaluate(results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP

