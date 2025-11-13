# python imports
import os
import glob
import time
import pickle
import argparse
from pprint import pprint

# torch imports
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# our code
from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.datasets import make_dataset, make_data_loader, to_frame_wise, to_segments
from libs.utils import valid_one_epoch, fix_random_seed

from taskgraph.task_graph_learning import (DO,
                                        extract_predecessors,
                                        delete_redundant_edges,
                                        load_config_task_graph_learning,
                                        sequences_accuracy)

################################################################################

# 顶部或 main 前添加：打印所有参数/缓冲的设备
def report_model_devices(model):
    root = model.module if hasattr(model, 'module') else model
    import torch
    print('[DEVICES] Parameters:')
    for name, p in root.named_parameters():
        print(f'  {name}: {p.device}')
    print('[DEVICES] Buffers:')
    for name, b in root.named_buffers():
        if isinstance(b, torch.Tensor):
            print(f'  {name}: {b.device}')

# 可选：只打印不在 cuda:4 的项（更聚焦）
def report_mismatched_devices(model, expected_device='cuda:1'):
    root = model.module if hasattr(model, 'module') else model
    bad = []
    for name, p in root.named_parameters():
        dev = getattr(p, 'device', None)
        if dev is not None and str(dev) != expected_device:
            bad.append(('param', name, dev))
    for name, b in root.named_buffers():
        dev = getattr(b, 'device', None)
        if dev is not None and str(dev) != expected_device:
            bad.append(('buffer', name, dev))
    print(f'[DEVICES] not on {expected_device}: {len(bad)}')
    for kind, name, dev in bad:
        print(f'  {kind}: {name} -> {dev}')


def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['test_split']) > 0, "Test set must be specified!"
    assert len(cfg['val_split']) > 0, "Validation set must be specified!"
    assert len(cfg['train_split']) > 0, "Train set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)


    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # model = model.to(cfg['devices'][0])
    
    # model.init_prototypes()

    # not ideal for multi GPU training
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])

    # GMMs = model.module.GMMs
    # for i,GMM in enumerate(GMMs) :
    #     print(f"GMM{i}.mu:{GMM.mu}")
    #     print(f"GMM{i}.var:{GMM.var}")

    del checkpoint
    train_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['train_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    train_loader = make_data_loader(
        train_dataset, False, None, 1, cfg['loader']['num_workers']
    )
    output_name = 'pred_seg_results_'


    print_freq = args.print_freq
    print("\nBuilding GMM Model")
    
    model.eval()
    
    # 检查是否存在已训练的GMM模型
    if model.module.gmm_exists(args.ckpt):
        print("Found existing GMM model, loading...")
        if model.module.gmm_load(args.ckpt):
            print("GMM model loaded successfully!")
        else:
            print("Failed to load GMM model, retraining...")
            # 重新训练GMM
            for iter_idx, video_list in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                with torch.no_grad():
                    
                    if iter_idx == 0:
                        model(video_list, mode='gmm_init')

                    model(video_list, mode='gmm')

                    if iter_idx == len(train_loader) - 1:
                        model(video_list, mode='gmm_fit')
            
            # 保存训练好的GMM模型
            model.module.gmm_save(args.ckpt)
    else:
        print("No existing GMM model found, training new model...")
        # 训练新的GMM模型
        for iter_idx, video_list in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            with torch.no_grad():
                
                if iter_idx == 0:
                    model(video_list, mode='gmm_init')
                # report_mismatched_devices(model, expected_device='cuda:1')
                # model = model.to()
                model(video_list, mode='gmm')

                if iter_idx == len(train_loader) - 1:
                    model(video_list, mode='gmm_fit')
        
        # 保存训练好的GMM模型
        model.module.gmm_save(args.ckpt)

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    test_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['test_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    test_loader = make_data_loader(
        test_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    # for threshold in range(0, 41):
    #     results = {
    #         'video-id': [],
    #         't-start' : [],
    #         't-end': [],
    #         'label': [],
    #         'score': []
    #     }
    #     output_file = os.path.join(args.ckpt, output_name+'%.2f.pkl'%(threshold))

    #     # loop over test set
    #     for iter_idx, video_list in enumerate(test_loader, 0):
    #         # forward the model (wo. grad)
    #         with torch.no_grad():
                
    #             # predict segments (boundaries and action steps)
    #             output = model(video_list)
    #             num_vids = len(output)
    #             for vid_idx in range(num_vids):
    #                 # generate frame-wise results and re-generate segments
    #                 preds = to_frame_wise(output[vid_idx]['segments'], output[vid_idx]['labels'],
    #                                     output[vid_idx]['scores'], video_list[vid_idx]['feats'].size(1), 
    #                                     fps=video_list[vid_idx]['fps'])
    #                 action_labels, time_stamp_labels = to_segments(preds)
    #                 video_id = output[vid_idx]['video_id']
    #                 video_list[vid_idx]['segments'] = torch.tensor(time_stamp_labels)
    #                 video_list[vid_idx]['labels'] = torch.tensor(action_labels).long()

    #             # perform error detection
    #             output = model(video_list, mode=args.mode, threshold=threshold)

    #             num_vids = len(output)
    #             for vid_idx in range(num_vids):
    #                 if output[vid_idx]['segments'].shape[0] > 0:
    #                     video_id = output[vid_idx]['video_id']
    #                     if video_id not in results:
    #                         results[video_id] = {}
    #                     results[video_id]['segments'] = output[vid_idx]['segments'].numpy()
    #                     results[video_id]['label'] = output[vid_idx]['labels'].numpy()
    #                     results[video_id]['score'] = output[vid_idx]['scores'].numpy()

    #         # printing
    #         if (iter_idx != 0) and iter_idx % (print_freq) == 0:
    #             torch.cuda.synchronize()
    #             print('Threshold:%.3f, Test: [%05d/%05d]\t'%(threshold, iter_idx, len(test_loader)))

    #     with open(output_file, "wb") as f:
    #         pickle.dump(results, f)
    
    all_results = {threshold: {} for threshold in range(0, 31)}

    # Loop over the test set
    for iter_idx, video_list in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc="Testing ED"):
        # Forward the model (without grad)
        with torch.no_grad():
            # Predict segments (boundaries and action steps)
            output = model(video_list)
            num_vids = len(output)
            
            # Generate frame-wise results and re-generate segments
            for vid_idx in range(num_vids):
                preds = to_frame_wise(output[vid_idx]['segments'], output[vid_idx]['labels'],
                                    output[vid_idx]['scores'], video_list[vid_idx]['feats'].size(1), 
                                    fps=video_list[vid_idx]['fps'])
                action_labels, time_stamp_labels = to_segments(preds)
                video_id = output[vid_idx]['video_id']
                video_list[vid_idx]['segments'] = torch.tensor(time_stamp_labels)
                video_list[vid_idx]['labels'] = torch.tensor(action_labels).long()

            # Perform error detection (new compute_prob version)
            threshold_outputs = model(video_list, mode=args.mode)

            # Process results for each threshold
            for threshold_result in threshold_outputs:  # `output` contains results for all thresholds (0-40)
                threshold = threshold_result['threshold']
                threshold_b_output = threshold_result['b_output']
                
                # Append results to the corresponding threshold in all_results
                for vid_idx in range(num_vids):
                    vid_output = threshold_b_output[vid_idx]
                    video_id = vid_output['video_id']
                    
                    if vid_output['segments'].shape[0] > 0:
                        # Initialize if this video_id is not already in the results for this threshold
                        if video_id not in all_results[threshold]:
                            all_results[threshold][video_id] = {}
                        
                        all_results[threshold][video_id]['segments'] = vid_output['segments'].numpy()
                        all_results[threshold][video_id]['label'] = vid_output['labels'].numpy()
                        all_results[threshold][video_id]['score'] = vid_output['scores'].numpy()

        # Printing progress
        # if (iter_idx != 0) and iter_idx % print_freq == 0:
        #     torch.cuda.synchronize()
        #     print(f'Test: [{iter_idx}/{len(test_loader)}] processed.')

    # After the entire test_loader has been processed, save all results for each threshold
    for threshold in range(0, 31):
        output_file = os.path.join(args.ckpt, output_name + '%.2f.pkl' % threshold)
        with open(output_file, "wb") as f:
            pickle.dump(all_results[threshold], f)
    
    
    print('Test set done!')

    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        help='print frequency (default: 20 iterations)')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--mode', default='prob', type=str)  
    parser.add_argument('--score', action='store_true')       
    args = parser.parse_args()
    main(args)

# online pcml
