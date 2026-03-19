model_name=llava_vid
dataset_name=videomme
frame_num=32
use_topk=True
base_score_path=./selected_frames/videomme/focus

python ./evaluation/change_score.py \
    --base_score_path $base_score_path \
    --dataset_name $dataset_name \
    --num_frames $frame_num

python ./evaluation/insert_frame_num.py \
    --frame_num $frame_num \
    --use_topk $use_topk

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=/kaggle/temp/LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=32,overwrite=False,use_topk=True,attn_implementation=sdpa \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llavavid_7b_qwen_lvb_v \
    --output_path /kaggle/working/results/${model_name}_focus_${dataset_name}_${frame_num}
