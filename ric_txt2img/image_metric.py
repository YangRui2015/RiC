import io
import os
import sys
import json
import numpy as np
from typing import Any
import uuid
import glob
import tqdm
import base64
import argparse
from io import BytesIO
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
# import pytorch_lightning as pl


import clip



IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


class MLP(torch.nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



def encode_jpeg(x, quality=95):
    """
    x : pil_img
    """
    img = x
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    jpeg = buffer.getvalue()
    return np.frombuffer(jpeg, dtype=np.uint8)


class ImageMetric():
    def __init__(self, rm1, rm2, device) -> None:
        self.device = device
        self.rm_names = [rm1, rm2]
        self.clip_model = None
        self.clip_preprocessor = None
        self.rms = [self.get_model(rm, device) for rm in [rm1, rm2]]

        # preprocess
        normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)
        self.colorAes_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])

    def get_model(self, rm, device):
        if rm == 'aesthetic_score':
            if self.clip_model is None and self.clip_preprocessor is None:
                self.clip_model, self.clip_preprocessor = clip.load("/mnt/aigc_cq/private/siboliu/Models/openai_clip/ViT-L-14.pt", device=device)
            model = MLP(input_size=768)
            s = torch.load("/mnt/aigc_cq/private/amandaaluo/own_code/AIGC/stable_diffusion_finetune/sd_train_1B_v1_multi_objective/pretrained/sac+logos+ava1-l14-linearMSE.pth") 
            model.load_state_dict(s, strict=True)
            model.to(device)
            model.eval()
            print('load aesthetic model')
            return model        
        elif rm == 'jpeg_size':
            return None
        else:
            NotImplementedError
            
    def get_image_features(self, image, model, preprocess):
        image = preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
        self.image_features = image_features

        return image_features

    def get_score(self, prompt, image):
        scores = []
        for i in range(2):
            rm_name = self.rm_names[i]
            if rm_name == 'aesthetic_score':
                score = self.get_aesthetic_score(self.rms[i], image)
            elif rm_name == 'jpeg_size':
                score = self.get_jpg_size(self.rms[i], image, prompt)
            
            scores.append(score)
                
        return scores[0], scores[1]

    def get_aesthetic_score(self, aes_model, image):
        with torch.no_grad():
            with torch.autocast(device_type='cuda'):
                image_features = self.get_image_features(image, self.clip_model, self.clip_preprocessor)
                score = aes_model(image_features)
        
        return score.item()

    def get_jpg_size(self, model, img_pil, caption):      
        length = len(encode_jpeg(img_pil))
        ## uint8 : 1 byte (8 bits) per dim
        sizes_kb = length / 1000.0
        
        return -sizes_kb


def convert_binary_pil(img_binary):
    img_bytes = BytesIO(img_binary)
    img_pil = Image.open(img_bytes)

    return img_pil


def generate_unique_id():
    unique_id = uuid.uuid4()

    return str(unique_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--data_idx",
        type=int,
        required=True,
        help="part index of data",
    )
    args = parser.parse_args()
    return args


