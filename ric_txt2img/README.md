# Ric in txt2img

## Evaluate
sh eval_jpeg_aes.sh

## Train
### Data preparation
1. Follow [here](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md) to download images and select a subset of them.
2. download model openai_clip/ViT-L-14.pt adn https://github.com/christophschuhmann/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth
3. Generate jsons and calculate attrbiutes like jpeg_size and aes score for samples.
python3 make_dataset.py --data_idx 0

4. merge all the generated json files.

### Start to train
sh train.sh


