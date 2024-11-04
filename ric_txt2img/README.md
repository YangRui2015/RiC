# Ric in txt2img

## Evaluate
Download model [here](https://huggingface.co/amandaa/ric_txt2img/tree/main) and put it to ./experiments
sh eval_jpeg_aes.sh

## Train
### Data preparation
We randomly sample English text-image pairs in LAION 5B(ie. Laion2B-en, because the dataset is multilingual).  Additionally, to mitigate the sparsity of high-quality images, we also randomly sample high-quality data with aesthetics scores of 6.5 or higher from LAION-Aesthetics.

1. Follow https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md and https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion-aesthetic.md to download images and select a subset of them.
2. download model openai_clip/ViT-L-14.pt and https://github.com/christophschuhmann/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth
3. Generate jsons and calculate attrbiutes like jpeg_size and aes score for samples, eg.
python3 make_dataset.py --data_idx 0

4. Merge all the generated json files.

### Start to train
sh train.sh


