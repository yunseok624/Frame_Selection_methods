model_name=llava_onevision
dataset_name=longvideobench
frame_num=32

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision fp16 --main_process_port 12345 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=/kaggle/temp/llava-onevision-qwen2-7b-ov,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=32,use_topk=False,attn_implementation=sdpa \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision_7b \
    --output_path /kaggle/working/results/${model_name}_uni_${dataset_name}_${frame_num}

