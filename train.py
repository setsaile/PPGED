##############################################################################################
# The code is modified from ActionFormer: https://github.com/happyharrycn/actionformer_release
##############################################################################################

# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our implementation
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch,train_one_epoch_gmm,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

def override_config_with_cmd_args(config, cmd_args):

    config['loader']['batch_size'] = cmd_args.batch_size
    # 可以在这里添加其他参数覆盖逻辑
    return config

################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)
    if args.batch_size is not None:
        cfg = override_config_with_cmd_args(cfg, args)
        print(f"batchsize modified to {args.batch_size}")
    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    if cfg['model']['train_cfg']['contrastive']:
        model.init_prototypes()


    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))

            # 获取模型的状态字典
            state_dict = checkpoint['state_dict']

            # 创建要删除的参数的列表
            # keys_to_remove = []
            # for i in range(14):  # 假设有 0 到 10 的 GMMs
            #     keys_to_remove.append(f'module.GMMs.{i}.mu')
            #     keys_to_remove.append(f'module.GMMs.{i}.var')
            #     keys_to_remove.append(f'module.GMMs.{i}.pi')

            # # 删除要跳过的参数
            # for key in keys_to_remove:
            #     if key in state_dict:
            #         del state_dict[key]

            # 加载更新后的状态字典
            model.load_state_dict(state_dict, strict=False)
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'], strict=False)
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint    
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    print('Maximum epoch:', max_epochs)
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer = tb_writer,
            print_freq = args.print_freq,
            use_contrastive = cfg['train_cfg']['contrastive'],
            batch_size = cfg['loader']['batch_size'],
            max_videos = args.maxvideos
        )

        # save ckpt once in a while
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            # if cfg['train_cfg']['gmm']:

            
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            if (epoch + 1) > int(max_epochs * 0.85):
                print('saving checkpoint...')
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
                )
            
    # train_one_epoch_gmm(
    #         train_loader,
    #         model,
    #         optimizer,
    #         scheduler,
    #         epoch+1,
    #         model_ema = model_ema,
    #         clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
    #         tb_writer = tb_writer,
    #         print_freq = args.print_freq,
    #         use_gmm = True,
    #         batch_size = cfg['loader']['batch_size'],
    #         max_videos = args.maxvideos
    # )
    # save_states = {
    #     'epoch': epoch + 2,
    #     'state_dict': model.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    # }
    # means = model.module.means_gmm
    # covariances = model.module.covariances_gmm
    # weights = model.module.weights_gmm
    # threshold = model.module.threshold_gmm
    # print(f"means:{means}")
    # print(f"covariances:{covariances}")
    # print(f"weights:{weights}")
    # print(f"threshold:{threshold}")
    # GMMs = model.module.GMMs
    # for i,GMM in enumerate(GMMs) :
    #     # print(f"GMM{i}.mu:{GMM.mu}")
    #     print(f"GMM{i}.mu.shape:{GMM.mu.shape}")
    #     print(f"GMM{i}.pi.shape:{GMM.pi}")
    #     # print(f"GMM{i}.var:{GMM.var}")
    #     print(f"GMM{i}.var.shape:{GMM.var.shape}")
    #     var_tensor = GMM.var  # 根据你的代码，获取 var tensor

    #     # 检查是否含有不是 0 或 1 的数字
    #     contains_non_zero_one = torch.any((var_tensor != 0) & (var_tensor != 1))

    #     if contains_non_zero_one:
    #         print("GMM.var 中包含不是 0 或 1 的数字。")
    #     else:
    #         print("GMM.var 中只包含 0 或 1。")

    # save_states['state_dict_ema'] = model_ema.module.state_dict()
    # print('saving checkpoint...')
    # save_checkpoint(
    #     save_states,
    #     False,
    #     file_folder=ckpt_folder,
    #     file_name='epoch_{:03d}.pth.tar'.format(epoch + 2)
    # )            
            
    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=2, type=int,
                        help='print frequency (default: 2 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--maxvideos', default=15, type=int,
                        help='number of training videos in each prototype (maxvideos * batch size)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--batch_size', type=int, help='Batch size for the data loader')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)