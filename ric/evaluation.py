import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, DataCollatorWithPadding
from peft import PeftModel
from trl import set_seed
import numpy as np
import pandas as pd
from utils import get_clean_data, Instructions_n, build_dataset_with_preference_n, \
                  load_main_tokenizer, Instructions_summary_n
from multi_reward_models import RewardModels
tqdm.pandas()
from utils import save_configs, map_rewards_from_preference, build_summary_dataset_with_preference_n, clean_gpu_memory

# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

@dataclass
class ScriptArguments:
    peft_name: Optional[str] = field(default='', metadata={'help': "path of the peft files"})
    num_prefer_points: Optional[int] = field(default=10)
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default='./logs_trl_eval/')
    wandb_name: Optional[str] = field(default='test', metadata={"help": "Name for this experiment"})
    reward_names:Optional[str] = field(default='harmless,helpful') 
    base_model_name: Optional[str] = field(default='meta-llama/Llama-2-7b-hf', metadata={"help": "local path to the base model or the huggingface id"})
    reward_stats_path: Optional[str] = field(default='')
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_model_name = script_args.base_model_name
tokenier_name = script_args.base_model_name
reward_stats_path = script_args.reward_stats_path if len(script_args.reward_stats_path) else None
print('base model: ', base_model_name)

peft_name = script_args.peft_name
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print(reward_names)
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
    

save_info = {
    'base_model_name': base_model_name,
    'peft_name': peft_name,
    'tokenier_name': tokenier_name
}
for i in range(len(reward_model_path_list)):
    save_info['reward_peft_path{}'.format(i+1)] = reward_model_path_list[i]
save_configs(save_info, os.path.join(script_args.save_directory, script_args.wandb_name))


process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

set_seed(8888)
tokenizer = load_main_tokenizer(tokenier_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.bfloat16,  # fast inference
    device_map=gpu_id, 
)
############# very important for padding
model.resize_token_embeddings(len(tokenizer))
if len(peft_name) > 0:
    model = PeftModel.from_pretrained(model, peft_name)
if hasattr(model, 'merge_and_unload'):
    model = model.merge_and_unload()

# load reward model
# do not normalization for evaluation
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id, reward_stats_path) 
num_rewards = len(reward_model_path_list)
instructions = Instructions_n(num_rewards) if exp_type == 'assistant' else Instructions_summary_n(num_rewards)

generation_kwargs = {
    "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9, 
    "do_sample": True,
}

### for evaluation
print('evaluation........')
tokenizer.padding_side = "left"

## preference list
if reward_models.num_rewards == 2:
    N = script_args.num_prefer_points
    preferences = np.zeros((N+1, 2))
    preferences[:, 0] = np.arange(0,1 + 1/N, 1/N)
    preferences[:, 1] = 1 - preferences[:, 0]
    preferences = np.round(preferences, 1)
elif reward_models.num_rewards == 3:
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
else: 
    raise NotImplementedError


# using Gaussian rewards as reference
rewards_reference_list = [np.random.randn(50000) for _ in range(len(preferences[0]))]

def evaluate_model(
    model, 
    reward_models,
    tokenizer, 
    target_rewards, 
    instructions, 
    gpu_id):
    if exp_type == 'assistant':
        valid_dataset = build_dataset_with_preference_n(hhrlhf_dataset_path, tokenizer, rm_tokenizers, target_rewards, split='test') 
    else:
        valid_dataset = build_summary_dataset_with_preference_n(summary_dataset_path, tokenizer, rm_tokenizers, target_rewards, split='test') 
    print(f"Size of the validation set: {len(valid_dataset)}")
    valid_batch_size = 1
    valid_dataset = valid_dataset.remove_columns('input_ids')
    valid_dataset = valid_dataset.rename_column('prompt_with_score_ids', 'input_ids')
    valid_dataset = valid_dataset.remove_columns(['prompt', 'response', 'query', 'score1', 'score2', 'prompt_with_score'])
    for key in ['key', 'text']:
        if key in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(key)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, collate_fn=data_collator)
    accelerator = Accelerator()
    model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

    full_response_tensors = []
    full_prompts = []
    pbar = tqdm(total=len(valid_dataset) // valid_batch_size // accelerator.num_processes)
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            response_tensors = accelerator.unwrap_model(model).generate(batch['input_ids'], **generation_kwargs)
            full_response_tensors.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)

    full_prompts = tokenizer.batch_decode(full_prompts)
    full_responses = tokenizer.batch_decode(full_response_tensors)
    # clean data
    full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

    # Compute score
    queries_responses = [(instructions.get_input(text),  instructions.get_response(text)) for text in full_responses]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses)

    desired_rewards_list = [[] for _ in range(reward_models.num_rewards)]
    for text in full_responses:
        desired_scores = instructions.get_scores(text)
        for i in range(reward_models.num_rewards):
            desired_rewards_list[i].append(float(desired_scores[i]))

    accelerator = Accelerator()
    accelerator.wait_for_everyone()
    ### merge data
    ### error here may because of old version of transformers/accelerate/peft
    all_rewards = []
    all_desired_rewards = []
    for i in range(reward_models.num_rewards):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
        all_desired_rewards.append(accelerator.gather_for_metrics(desired_rewards_list[i]))
    all_full_prompts = accelerator.gather_for_metrics(full_prompts)
    all_full_responses = accelerator.gather_for_metrics(full_responses)
    return all_rewards, all_desired_rewards, all_full_prompts, all_full_responses


rm_tokenizers = []
for i in range(num_rewards):
    rm_tokenizers.append(AutoTokenizer.from_pretrained(rm_tokenizer_path_list[i]))
    
for k in range(len(preferences)): 
    preference = preferences[k]
    target_rewards = map_rewards_from_preference(rewards_reference_list, preference, method='l2').reshape(-1)
    print(k, target_rewards)
    all_rewards, all_desired_rewards, all_full_prompts, all_full_responses = evaluate_model(model, reward_models, tokenizer, target_rewards, instructions, gpu_id)
    if process_id == 0:
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
        }
        for i in range(num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            evaluation_result['desired_score{}'.format(i+1)] = all_desired_rewards[i]

            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))
            print('total average desired score {}: {}'.format(i+1, np.mean(evaluation_result['desired_score{}'.format(i+1)])))

        dataframe = pd.DataFrame(evaluation_result)
        if len(preference) == 3:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}_{}.csv'.format(preference[0], preference[1], preference[2])))
        else:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}.csv'.format(preference[0], preference[1])))
        


