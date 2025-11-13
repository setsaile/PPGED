#!/bin/bash

input_dir="./data/EgoPER/coffee/trim_videos"
output_dir="./data/EgoPER/coffee/frames_10fps"

process_video() {
    filename=$1
    id="${filename%.*}"
    
    # 检查是否已经存在目标文件夹
    if [ -d "$output_dir/$id" ]; then
        echo "Skipping $filename, frames already extracted."
    else
        mkdir -p "$output_dir/$id"
        ffmpeg -i "$input_dir/$filename" -vf "fps=10" "$output_dir/$id/%06d.png"
    fi
}

export -f process_video
export input_dir output_dir

# 使用 ls 列出文件并通过 xargs 并行处理
ls $input_dir | xargs -I {} -P 32 bash -c 'process_video "$@"' _ {}
