model_name=qwen2_vl
dataset_name=longvideobench
frame_num=32
base_score_path=./selected_frames/longvideobench/clip

python ./evaluation/change_score.py \
    --base_score_path $base_score_path \
    --dataset_name $dataset_name \
    --num_frames $frame_num

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision fp16 --main_process_port 12345 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=/kaggle/temp/Qwen2-VL-7B-Instruct,use_topk=True,nframes=32 \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_vl_7b \
    --output_path /kaggle/working/results/${model_name}_aks_${dataset_name}_${frame_num}