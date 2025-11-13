#!/bin/bash

export PYTHONPATH=`pwd`
root_dir=./data/EgoPER/coffee

video_dir=$root_dir/frames_10fps
feature_dir=$root_dir/features_10fps

# 定义提取特征的函数
extract_features() {
    video=$1
    index=$2
    video_id="${video%.*}"  # 去除扩展名，获取视频ID
    feature_file="$feature_dir/$video_id.npy"
    
    # 根据任务索引决定使用哪个GPU
    if [ $((index % 2)) -eq 0 ]; then
        gpu=0
    else
        gpu=1
    fi

    if [ -f "$feature_file" ]; then
        echo "Features for $video already exist, skipping."
    else
        echo "Processing $video on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python -m src.feature_extract \
            --feature_model src/feature_extractor/pretrained_models/kinetics400-rgb-i3d-resnet-50-f32-s2-precise_bn-warmupcosine-bs1024-e196.pth.tar \
            --frames $video_dir/$video \
            --savedir $feature_dir \
            --mp
    fi
}

export -f extract_features
export video_dir feature_dir

# 获取视频文件列表
videos=($(ls $video_dir))

# 使用带索引的循环来处理每个视频
for i in "${!videos[@]}"; do
    echo "${videos[$i]}" $i
done | xargs -n 2 -P 64 bash -c 'extract_features "$@"' _ 
