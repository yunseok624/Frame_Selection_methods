base_score_path=./selected_frames/longvideobench/clip
score_type=selected_frames
dataset_name=longvideobench
frame_num=32

python ./evaluation/change_score.py \
    --base_score_path $base_score_path \
    --score_type $score_type \
    --dataset_name $dataset_name \
    --num_frames $frame_num

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=/content/drive/MyDrive/checkpoints/LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=32,overwrite=False,use_topk=True,attn_implementation=sdpa\
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llavavid_7b_qwen_lvb_v \
    --output_path /content/drive/MyDrive/results/${dataset_name}_aks_${frame_num}