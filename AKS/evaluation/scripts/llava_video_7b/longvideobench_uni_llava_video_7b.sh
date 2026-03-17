score_type=longvideobench_uni
frame_num=32

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision fp16 --main_process_port 12345 -m lmms_eval\
    --model llava_vid \
    --model_args pretrained=/kaggle/temp/LLaVA-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=32,overwrite=False,use_topk=False,attn_implementation=sdpa\
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llavavid_7b_qwen_lvb_v \
    --output_path /kaggle/working/results/${score_type}_${frame_num}
