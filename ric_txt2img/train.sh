export HOST_NUM=1
export INDEX=0
export CHIEF_IP=localhost
export HOST_GPU_NUM=8

PROCESS_NUM=$((HOST_GPU_NUM * HOST_NUM))
echo ${PROCESS_NUM}
export NCCL_IB_DISABLE=1

export MODEL_NAME="/mnt/aigc_cq/shared/txt2img_models/stable-diffusion-v1-5"
export TRAIN_JSON="/mnt/aigc_cq/private/amandaaluo/dataset/multi_obj/multi_obj_with_clip_base.json"


accelerate launch --mixed_precision="fp16" --gpu_ids 0,1,2,3,4,5,6,7 \
  --use_deepspeed --num_processes ${PROCESS_NUM} \
  --deepspeed_config_file ./configs/zero2_config.json \
  --num_machines ${HOST_NUM} --machine_rank ${INDEX} --main_process_ip ${CHIEF_IP} --main_process_port 2006 \
  --deepspeed_multinode_launcher standard \
  examples/text_to_image/train_multi_obj_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_json=$TRAIN_JSON \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=8 \
  --dataloader_num_workers=8 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=15000000 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --enable_xformers_memory_efficient_attention \
  --output_dir="experiments/multi_obj_aes_jpg_0126" \
  --rm1 "aesthetic_score" \
  --rm2 "jpeg_size"
