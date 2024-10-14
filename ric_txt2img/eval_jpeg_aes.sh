# multi-gpu inference

steps=(400000) 
for ((i=0; i<${#steps[@]}; i++))
do  
    step=${steps[$i]}
    unet_model_name_or_path=experiments/multi_obj_aes_jpg_0126/checkpoint-${step}/unet
    rm1="aesthetic_score" # "aesthetic_score" # "aesthetic_score" # "color_aes" # "jpeg_size" 
    rm2="jpeg_size" # "clip_large_score" # "RM" # "jpeg_size" # "RM"

    for i in {0..7}
    do
        python3 examples/text_to_image/infer_text_to_image_with_preference_batch.py --unet_model_name_or_path ${unet_model_name_or_path} --rm1 ${rm1} --rm2 ${rm2} --device ${i} --preference_idx ${i} &
    done
    wait

    num_node=1
    for i in {0..2}
    do
        python3 examples/text_to_image/infer_text_to_image_with_preference_batch.py --unet_model_name_or_path ${unet_model_name_or_path} --rm1 ${rm1} --rm2 ${rm2} --device ${i} --preference_idx `expr $num_node \* 8 + $i` &
    done
    wait

    device=0
    python3 examples/text_to_image/evaluate.py --rm1 ${rm1} --rm2 ${rm2} --device ${device} --step ${step}
done

