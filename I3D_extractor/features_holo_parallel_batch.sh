#!/bin/bash

export PYTHONPATH=$(pwd)
root_dir=./data/HoloAssist

video_dir=$root_dir/frames_10fps
feature_dir=$root_dir/features_10fps

# 定义提取特征的函数
extract_features() {
    video="$1"
    index=$2
    video_id="${video%.*}"  # 去除扩展名，获取视频ID
    feature_file="$feature_dir/$video_id.npy"
    
    # 根据任务索引决定使用哪个GPU
    if [ $((index % 3)) -eq 0 ]; then
        gpu=3
    elif [ $((index % 3)) -eq 1 ]; then
        gpu=4
    else
        gpu=5
    fi

    if [ -f "$feature_file" ]; then
        echo "Features for $video already exist, skipping."
    else
        echo "Processing $video on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python -m src.feature_extract_batch \
            --feature_model src/feature_extractor/pretrained_models/kinetics400-rgb-i3d-resnet-50-f32-s2-precise_bn-warmupcosine-bs1024-e196.pth.tar \
            --frames "$video_dir/$video" \
            --savedir "$feature_dir" \
            --mp
    fi
}

export -f extract_features
export video_dir feature_dir

# 获取视频文件列表并去除已存在的特征文件
videos=($(ls "$video_dir"))
videos_to_process=()

for video in "${videos[@]}"; do
    video_id="${video%.*}"  # 去除扩展名，获取视频ID
    feature_file="$feature_dir/$video_id.npy"
    if [ ! -f "$feature_file" ]; then
        videos_to_process+=("$video")  # 仅保留未处理的视频
    else
        echo "Features for $video already exist, skipping."
    fi
done

# 计算要平分的视频数量
num_videos=${#videos_to_process[@]}
third=$((num_videos / 3))

# 将视频平分到三个数组
videos_gpu3=("${videos_to_process[@]:0:$third}")
videos_gpu4=("${videos_to_process[@]:$third:$third}")
videos_gpu5=("${videos_to_process[@]:$((2 * $third))}")

# 启动三个并行任务
(
    for video in "${videos_gpu3[@]}"; do
        extract_features "$video" 0
    done
) &

(
    for video in "${videos_gpu4[@]}"; do
        extract_features "$video" 1
    done
) &

(
    for video in "${videos_gpu5[@]}"; do
        extract_features "$video" 2
    done
) &

wait  # 等待所有后台进程完成
