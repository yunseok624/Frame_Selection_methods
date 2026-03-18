model_name=qwen2_vl
dataset_name=videomme
frame_num=32
use_topk=True
base_score_path=./selected_frames/videomme/clip

python ./evaluation/change_score.py \
    --base_score_path $base_score_path \
    --dataset_name $dataset_name \
    --num_frames $frame_num

python ./evaluation/insert_frame_num.py \
    --frame_num $frame_num \
    --use_topk $use_topk

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision fp16 --main_process_port 12345 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=./checkpoints/Qwen2-VL-7B-Instruct,use_topk=True,nframes=32 \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_vl_7b \
    --output_path /kaggle/working/results/${model_name}_aks_${dataset_name}_${frame_num}

