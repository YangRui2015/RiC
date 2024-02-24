import os
import gc
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, DataCollatorWithPadding
from torch.utils.data import DataLoader
from trl import set_seed
import numpy as np
import pandas as pd
from utils import get_clean_data, load_main_tokenizer, save_configs, print_trainable_parameters, \
                  merge_weights_with_preference, Instructions, Instructions_summary, build_dataset_eval, build_dataset_summary_eval
from multi_reward_models import RewardModels
tqdm.pandas()


# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'
tokenizer_path = 'meta-llama/Llama-2-7b-hf'

@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_rewardedsoups_summary_eval')
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "minibatch size for eval"})
    wandb_name: Optional[str] = field(default='eval_pposoups_klreg0.2_harmless_helpful', metadata={"help": "Name for this experiment"})
    reward_names:Optional[str] = field(default='harmless,helpful')
    base_model_path1: Optional[str]=field(default='./ppo_llamma2_klreg0.2_harmless/batch_200')
    base_model_path2: Optional[str]=field(default='./ppo_llamma2_klreg0.2_helpful/batch_200')
    base_model_path3: Optional[str]=field(default='')
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_model_name_1 = script_args.base_model_path1
base_model_name_2 = script_args.base_model_path2
base_model_name_3 = script_args.base_model_path3
tokenier_name = tokenizer_path

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['mohameddhiab/humor-no-humor'],
}
reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])
os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)
save_info = {
    'base_model_name_1': base_model_name_1,
    'base_model_name_2': base_model_name_2,
    'base_model_name_3': base_model_name_3,
    'reward_peft_path1': reward_model_path_list[0],
    'reward_peft_path2': reward_model_path_list[1],
    'tokenier_name': tokenier_name
}
for i in range(len(reward_model_path_list)):
    save_info['reward_peft_path{}'.format(i+1)] = reward_model_path_list[i]
save_configs(save_info, os.path.join(script_args.save_directory, script_args.wandb_name))


accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id) 


set_seed(8888)
current_device = Accelerator().local_process_index
print(current_device)


tokenizer = load_main_tokenizer(tokenier_name)
if exp_type == 'assistant':
    valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test') 
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions_summary()
print(f"Size of the validation set: {len(valid_dataset)}")

def evaluate_model(temp_save_path, tokenizer, valid_dataset):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, batch_size=script_args.mini_batch_size, drop_last=True, collate_fn=data_collator)
    ### load_merged_model
    model = AutoModelForCausalLM.from_pretrained(
        temp_save_path,
        torch_dtype=torch.bfloat16, # for merging model
        device_map=gpu_id
        )
    model.resize_token_embeddings(len(tokenizer))

    accelerator = Accelerator()
    model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

    generation_kwargs = {
        "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9, 
        "do_sample": True,
    }

    full_responses = []
    full_prompts = []
    pbar = tqdm(total=len(valid_dataset) // script_args.mini_batch_size // accelerator.num_processes)
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            response_tensors = accelerator.unwrap_model(model).generate(batch["input_ids"], **generation_kwargs) #length_sampler=output_length_sampler, 
            full_responses.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)
    
    full_responses = tokenizer.batch_decode(full_responses)
    full_prompts = tokenizer.batch_decode(full_prompts)
    # clean data
    full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

    queries_responses = [
        (instructions.get_input(text), instructions.get_response(text))
        for text in full_responses
    ]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses)
    
    ### error here may because of old version of transformers/accelerate/peft
    all_rewards = []
    for i in range(reward_models.num_rewards):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
    all_full_prompts = accelerator.gather_for_metrics(full_prompts)
    all_full_responses = accelerator.gather_for_metrics(full_responses)
    return all_rewards, all_full_prompts, all_full_responses


print("Evaluating........")
tokenizer.padding_side = "left"
## preference list
if reward_models.num_rewards == 3:
    preferences = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.1, 0.8],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.4, 0.4],
        [0.2, 0.6, 0.2],
        [0.33, 0.33, 0.33],
        [0.4, 0.4, 0.2],
        [0.4, 0.2, 0.4], 
        [0.6, 0.2, 0.2],
        [0.8, 0.1, 0.1], 
        [1.0, 0.0, 0.0], 
        ])
elif reward_models.num_rewards == 2:
    M = 10
    preferences = np.zeros((M+1, 2))
    preferences[:, 0] = np.arange(0,1+ 1/M,1 / M)
    preferences[:, 1] = 1 - preferences[:, 0]
    preferences = np.round(preferences, 1)
else:
    raise NotImplementedError

for k in range(0, len(preferences)):
    preference = preferences[k]
    temp_save_path = './temp_models/{}'.format(script_args.wandb_name)
    if process_id == 0:
        if len(preference) == 3:
            base_model_list = [base_model_name_1, base_model_name_2, base_model_name_3]
        else:
            base_model_list = [base_model_name_1, base_model_name_2]
        merge_weights_with_preference(base_model_list, preference, temp_save_path)

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()
    print(k, preference)
    all_rewards, all_full_prompts, all_full_responses = evaluate_model(temp_save_path, tokenizer, valid_dataset)
    gc.collect()
    torch.cuda.empty_cache()

    if process_id == 0:
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
        }
        for i in range(reward_models.num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))

        dataframe = pd.DataFrame(evaluation_result)
        if len(preference) == 2:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}.csv'.format(preference[0], preference[1])))
        else:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}_{}.csv'.format(preference[0], preference[1], preference[2])))




