import json
import random
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


from transformers import CLIPTokenizer




class Multi_Obj_Dataset(Dataset):
    def __init__(self, json_file, args, rm1="aesthetic_score", rm2="clip_large_score") -> None:
        self.data = json.load(open(json_file))
        self.rm1 = rm1
        self.rm2 = rm2
        print("rm1:", rm1)
        print("rm2:", rm2)


        rm1_scores = np.array([item[rm1] for item in self.data])
        self.rm1_mean = np.mean(rm1_scores)
        self.rm1_std = np.std(rm1_scores)
        print(f"{rm1} mean:", self.rm1_mean)
        print(f"{rm1} std:", self.rm1_std)
        print(f"{rm1} max:", max(rm1_scores))
        print(f"{rm1} min:", min(rm1_scores))

        rm2_scores = np.array([item[rm2] for item in self.data])
        self.rm2_mean = np.mean(rm2_scores)
        self.rm2_std = np.std(rm2_scores)
        print(f"{rm2} mean:", self.rm2_mean)
        print(f"{rm2} std:", self.rm2_std)
        print(f"{rm2} max:", max(rm2_scores))
        print(f"{rm2} min:", min(rm2_scores))

        print("corr:", np.corrcoef(rm1_scores, rm2_scores))

        self.trans = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
        self.tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
        # clip_tokenizer_model_path = "/mnt/aigc_cq/private/huye/model_weights/stable-diffusion-v1-5"
        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #     clip_tokenizer_model_path, subfolder="tokenizer"
        # )
        self.drop_rate = 0.1


    def __len__(self):
        return len(self.data)

    
    def tokenize_caption(self, idx):
        text = self.data[idx]["caption"]

        rnd_value = random.random()
        if rnd_value < self.drop_rate:
            text = ""
            # reward_info = f"<rm1> null" # 
            aes_info, clip_info = "", ""
        else:
            # r1:
            rm1_score = (self.data[idx][self.rm1] - self.rm1_mean) / self.rm1_std
            rm1_score = str(round(rm1_score, 1))
            aes_info = f"<rm1> {rm1_score}"

            # r2:
            rm2_score = (self.data[idx][self.rm2] - self.rm2_mean) / self.rm2_std
            rm2_score = str(round(rm2_score, 1))
            clip_info = f"<rm2> {rm2_score}"
        
            print(rm1_score, rm2_score)

        score_len = 8
        total_score_len = score_len * 2
        
        text_inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length-total_score_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        eos_idx = sum(text_inputs.attention_mask[0])
        truncate_text = self.tokenizer.decode(text_inputs.input_ids[0][1:eos_idx-1])

        ## Tokenize reward info
        # r1
        aes_inputs = self.tokenizer(
            aes_info,
            max_length=score_len+2,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        eos_idx = sum(aes_inputs.attention_mask[0])
        truncate_aes = self.tokenizer.decode(aes_inputs.input_ids[0][1:eos_idx-1])

        # r2
        clip_inputs = self.tokenizer(
            clip_info,
            max_length=score_len+2,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        eos_idx = sum(clip_inputs.attention_mask[0])
        truncate_clip = self.tokenizer.decode(clip_inputs.input_ids[0][1:eos_idx-1])
        

        if truncate_text == "" and truncate_aes == "" and truncate_clip == "":
            concat_text = ""
        else:
            concat_text = (truncate_text + " " + truncate_aes + " " + truncate_clip)

        # print('concat_text')
        # print(concat_text)

        concat_inputs = self.tokenizer(
            concat_text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        real_concat_text = self.tokenizer.decode(concat_inputs.input_ids[0])

        print(real_concat_text)
        
        return concat_inputs


    def __getitem__(self, idx):
        item = self.data[idx]
        # read image
        image = Image.open(item["img_name"]).convert("RGB")
        image = self.trans(image)
        
        # tokenize caption
        concat_inputs = self.tokenize_caption(idx)

        sample = {}
        sample["input_ids"] = concat_inputs.input_ids
        sample["pixel_values"] = image
        
        return sample


if __name__ == "__main__":
    train_json_file="/mnt/aigc_cq/private/amandaaluo/dataset/multi_obj/multi_obj_with_jpeg_neg.json" 
    args = type('Args', (), {
    "resolution": 512,
    "center_crop": True,
    "random_flip": False,
    "revision": None,
    "pretrained_model_name_or_path": "/mnt/aigc_cq/shared/txt2img_models/stable-diffusion-v1-5",
})()
    
    dataset = Multi_Obj_Dataset(
        train_json_file, args, rm1="aesthetic_score", rm2="jpeg_size")

    for data in dataset:
        pass
        # import pdb;pdb.set_trace()
