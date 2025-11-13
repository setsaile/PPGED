import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import logging
try:
    from .datasets import register_dataset
    from .data_utils import truncate_feats
except ImportError:
    from datasets import register_dataset
    from data_utils import truncate_feats
import pickle
from collections import defaultdict
import heapq

@register_dataset("CaptainCook4D")
class captainCook4D(Dataset):
    def __init__(
        self,
        is_training,        # if in training mode
        split,              # split, a tuple/list allowing concat of subsets
        max_seq_len,        # maximum sequence length during training
        trunc_thresh,       # threshold for truncate an action segment
        crop_ratio,         # a tuple (e.g., (0.9, 1.0)) for random cropping
        task,               # task name
        default_fps = 10,
        features_subdir = 'I3D',
        **kwargs
    ):
        root_dir = ''
        self.feat_root_dir = os.path.join(root_dir, 'features_10fps')
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.task = task

        # 首先读取test.txt文件获取要处理的视频列表
        test_file_path = os.path.join(root_dir, f'{self.split}.txt')
        if os.path.exists(test_file_path):
            with open(test_file_path, 'r') as f:
                self.video_list = [line.strip() for line in f if line.strip()]
            print(f"从 {test_file_path} 加载了 {len(self.video_list)} 个视频")
        else:
            self.video_list = None
            print(f"未找到 {test_file_path}，将处理所有视频")

        if self.split == 'training':
            annotation_path = os.path.join(root_dir, 'non_error_samples_processed.json')
        elif self.split == 'test':
            annotation_path = os.path.join(root_dir, 'error_samples_processed.json')
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        with open(annotation_path, 'r') as f:
            self.annotation_data = json.load(f)[self.task]

        self.annots = []
        for video_id, video_data in self.annotation_data.items():
            # 如果有video_list，只处理列表中的视频
            if self.video_list is not None:
                # 对于task='0'，需要处理video_id格式
                if self.task == '0':
                    temp_video_id = video_id.split('_', 1)[1]
                    if temp_video_id not in self.video_list:
                        continue
                else:
                    if video_id not in self.video_list:
                        continue
            
            # feat
            if self.task != '0':
                feat_path = os.path.join(self.feat_root_dir, f'{video_id}.npy')
            else:
                temp_video_id = video_id.split('_', 1)[1]
                feat_path = os.path.join(self.feat_root_dir, f'{temp_video_id}.npy')
            
            # 检查特征文件是否存在
            if not os.path.exists(feat_path):
                continue
        
            # segments
            segments = video_data['segments']

            # labels
            labels = video_data['labels']

            # labels_error
            labels_error = video_data['labels_error']

            # descriptions
            descriptions = video_data['descriptions']

            self.annots.append({
                'video_id': video_id,
                'feat_path': feat_path,
                'segments': segments,
                'labels': labels,
                'labels_error': labels_error,
                'descriptions': descriptions
            })

        print(f"最终加载了 {len(self.annots)} 个有效视频样本")

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        annot = self.annots[idx]

        video_id = annot['video_id']
        feat_path = annot['feat_path']
        feats = np.load(feat_path) # Shape: (T, 2048)

        segments = annot['segments']
        labels = annot['labels']
        labels_error = annot['labels_error']
        descriptions = annot['descriptions']
        # to tensor
        segments = torch.tensor(segments).float()
        labels = torch.tensor(labels).long()
        labels_error = torch.tensor(labels_error).long()

        data_dict = {
            'video_id': video_id,
            'feats': torch.from_numpy(feats).transpose(0, 1).float(),
            'segments': segments,
            'labels': labels,
            'labels_error': labels_error,
            'fps': self.default_fps,
            'duration': len(feats) / self.default_fps,
        }

        # truncate the features during training
        if self.is_training:
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)

        return data_dict



# if __name__ == '__main__':
#     # 为每个任务处理序列
#     task='1'
#     dataset = captainCook4D(
#         is_training=True,
#         split='train',
#         max_seq_len=2304,
#         trunc_thresh=0.5,
#         crop_ratio=[0.9, 1.0],
#         task=task,
#     )

#     print(len(dataset))
#     for sample in dataset:
#         print(sample)
#         assert False