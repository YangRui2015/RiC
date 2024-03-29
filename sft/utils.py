import os
import gc
import copy
import shutil
from dataclasses import dataclass, field
from typing import Optional
from peft import PeftModel
from accelerate import Accelerator
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset, disable_caching
import numpy as np
import pandas as pd
from tqdm import tqdm
disable_caching()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_rewards(reward_model, texts_for_rewards, reward_mean_std=None, sub_position=0):
    rewards = []
    print('log: reward model forwarding ...')
    with torch.no_grad():
        pbar = tqdm(total=len(texts_for_rewards))
        for inputs in texts_for_rewards:
            if sub_position != 0: # for multiple output
                rewards.append(reward_model(**(inputs.to(reward_model.device))).logits[0][sub_position])
            else:
                rewards.append(reward_model(**(inputs.to(reward_model.device))).logits[0])
            pbar.update(1)
    
    if reward_mean_std is None:
        rewards = [np.round(r.cpu().detach().item(), 1) for r in rewards]
    else:
        mean_reward, std_reward = reward_mean_std
        rewards = [np.round((r.cpu().detach().item() - mean_reward) / std_reward, 1) for r in rewards]
    return rewards


def load_reward_model(reward_peft_path, gpu_id):
    num_labels = 2 if ('humor' in reward_peft_path or 'faithful' in reward_peft_path) else 1
    reward_model = AutoModelForSequenceClassification.from_pretrained(
                    reward_peft_path,
                    num_labels=num_labels, torch_dtype=torch.bfloat16,
                    device_map=gpu_id,
                    )
    if check_lora_in_model_path(reward_model, reward_peft_path):
        reward_model = PeftModel.from_pretrained(reward_model, reward_peft_path)
    if hasattr(reward_model, 'merge_and_unload'):
        reward_model = reward_model.merge_and_unload() # merge lora weights
    return reward_model.to(gpu_id)


def load_main_tokenizer(tokenier_name):
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>" 
    DEFAULT_UNK_TOKEN = "<unk>" 

    tokenizer = AutoTokenizer.from_pretrained(tokenier_name, use_fast = False)
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tokenizer

class Instructions:
    response_split = "\n\nAssistant:"
    input_split = "\n\nHuman:"
    score_split = "<rm_score>"

    @staticmethod
    def get_input(query):
        if Instructions.score_split in query:
            before_response = query.split(Instructions.score_split)[0]
        else:
            before_response = Instructions.response_split.join(query.split(Instructions.response_split)[:-1])
        return before_response.rstrip() + ' ' + Instructions.response_split
        
    @staticmethod
    def get_response(response):
        return response.split(Instructions.response_split)[-1].strip()

class Instructions_summary():
    instruction_summary = "Generate a one-sentence summary of this post."
    response_split = "### Response:"
    input_split = "### Input:"
    instruction_split = "### Instruction:"

    @classmethod
    def prompt_input(self, input):
        # formulate the news
        return f"### Instruction: {Instructions_summary.instruction_summary} ### Input: {input} ### Response: "

    def get_prompt(self, query):
        before_response = self.response_split.join(query.split(self.response_split)[:-1])
        return before_response.rstrip() 

    def get_post(self, query):
        before_response = self.get_prompt(query)
        return before_response.split(self.input_split)[1].strip()

    def get_input(self, query):
        return self.get_prompt(query) + ' ' + self.response_split
        
    def get_response(self, response):
        return response.split(self.response_split)[-1].strip()



def build_dataset(path, tokenizer, split='train', size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        sample['text'] = sample['chosen'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["text"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds_chosen = ds.map(tokenize, batched=False, num_proc=30)
    ds_concat = ds_chosen
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    ds_concat = ds_concat.remove_columns(['chosen', 'rejected'])

    ds_concat.set_format(type="torch")
    return ds_concat


def build_dataset_summary(path, tokenizer, split='train', size=None):
    ds = load_dataset(path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=20)

    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        choice = sample["choice"] # select the best summary
        sample["response"] = sample["summaries"][choice]["text"].replace("\n", " ").strip()
        sample["input_ids"] = tokenizer.encode(prompt_summary + sample["response"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds



def build_dataset_eval(path, tokenizer,rm_tokenizer1,rm_tokenizer2, split='test', size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))
    ds = ds.select(range(0, len(ds), 4))

    def tokenize(sample):
        sample['text'] = sample['chosen'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["input_ids_rm1"] = rm_tokenizer1.encode(sample["prompt"])
        sample["input_ids_rm2"] = rm_tokenizer2.encode(sample["prompt"])
        return sample

    ds_chosen = ds.map(tokenize, batched=False, num_proc=20)
    ds_concat = ds_chosen
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8 \
                                 and len(x["input_ids_rm1"]) <= 512 and len(x["input_ids_rm1"]) >= 8
                                 and len(x["input_ids_rm2"]) <= 512 and len(x["input_ids_rm2"]) >= 8 )
    ds_concat = ds_concat.remove_columns(['chosen', 'rejected','input_ids_rm1', 'input_ids_rm2'])

    ds_concat.set_format(type="torch")
    return ds_concat

def build_dataset_summary_eval(path, tokenizer, rm_tokenizer1,rm_tokenizer2, split='test', size=None):
    if split == 'test':
        split = 'validation'
    ds = load_dataset(path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=30)

    # need to remove duplicated prompts for evaluation
    def remove_duplicate(duplicated_dataset):
        duplicated_dataset = duplicated_dataset.filter(lambda x: x['info']["id"] is not None)
        initial_list = duplicated_dataset.map(lambda x: {"id": x['info']["id"]})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds = remove_duplicate(ds)
    if size is not None:
        ds = ds.select(range(size))
    ds = ds.select(range(0, min(len(ds),2000))) # select 2000 data 

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        sample["input_ids"] = tokenizer.encode(prompt_summary)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


def check_lora_in_model_path(model, path):
    if os.path.exists(path):
        dirnames = os.listdir(path)
        if 'adapter_config.json' in dirnames:
            return True
        
        state_dict_keys = model.state_dict().keys()
        for key in state_dict_keys:
            if 'lora' in key:
                return True
    return False

def get_clean_data(full_responses, full_prompts):
    full_responses_clean = []
    for i, response in enumerate(full_responses):
        full_prompts[i] = full_prompts[i].strip('[PAD] ')
        response = response.strip('[PAD] ')
        temp_resp = response.replace(full_prompts[i], '').strip('<s>').strip('</s>')
        if '</s>' in temp_resp:
            temp_resp = temp_resp[:temp_resp.rindex('</s>')]
        temp_resp = temp_resp.split('\n\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\n\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('###')[0].strip()
        temp_resp = temp_resp.split('\n\n\n')[0].strip()
        full_responses_clean.append(full_prompts[i] + ' ' + temp_resp)
    return full_responses_clean