import os
import copy
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from utils import clean_gpu_memory, merge_dataset, save_configs
from training import train_model
from generation import generate_data
from transformers import HfArgumentParser

# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    disable_wandb: Optional[str] = field(default=False, metadata={'help': 'Whether to disable wandb or not.'})
    save_directory: Optional[str] = field(default='./logs_trl/', metadata={'help':'path'})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    training_steps: Optional[int] = field(default=20000, metadata={'help': 'number of training steps in the offline training'})
    online_training_steps: Optional[int] = field(default=4000, metadata={'help': 'number of training steps in the online training'})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    num_online_iterations: Optional[int] = field(default=1, metadata={'help': 'number of the online generation and training'})
    num_generation_samples: Optional[int] = field(default=20000, metadata={'help': 'number of samples generated'})
    max_grad_norm: Optional[float] = field(default=1, metadata={"help": "Maximum gradient norm for gradient clipping"})
    quantile_threshold: Optional[float] = field(default=0.7)
    num_origin_samples: Optional[int] = field(default=10000) 
    wandb_name: Optional[str] = field(default='ric_assistant_harmlesshelpful_offline20000_lr1e-4', metadata={"help": "Name for this experiment"})
    base_model_name: Optional[str] = field(default='meta-llama/Llama-2-7b-hf', metadata={"help": "local path to the base model or the huggingface id"})
    reward_names:Optional[str] = field(default='harmless,helpful') 
    train_dataset_path: Optional[str] = field(default='./datasets/all_full_train_harmhelp.hf')
    train_reward_stats_path: Optional[str] = field(default='')
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_model_name = script_args.base_model_name
tokenizer_name = script_args.base_model_name
print('base model: ', base_model_name)

if script_args.disable_wandb: # if you don't need the wandb log
    os.environ['WANDB_DISABLED'] = 'true' 

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

# Remember to generate these datasets before training
train_dataset_path = script_args.train_dataset_path
reward_stats_path = script_args.train_reward_stats_path if len(script_args.train_reward_stats_path) else script_args.train_dataset_path + '/all_reward_stat.npy'


save_info = {
    'train_dataset_path': train_dataset_path,
    'base_model_name': base_model_name,
    'tokenizer_name': tokenizer_name
}
for i in range(len(reward_model_path_list)):
    save_info['reward_peft_path{}'.format(i+1)] = reward_model_path_list[i]
save_configs(save_info, os.path.join(script_args.save_directory, script_args.wandb_name))

save_path = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(save_path, exist_ok=True)

## offline training 
dataset = train_model(
    base_model_name=base_model_name,
    reward_model_path_list=reward_model_path_list,
    train_dataset=train_dataset_path,
    save_path=save_path + '/model_iter0',
    tokenizer_name=tokenizer_name,
    rm_tokenizer_path_list=rm_tokenizer_path_list,
    training_steps=script_args.training_steps,
    learning_rate=1.414e-4,  # use larger lr for offline training than  online training (script_args.learning_rate)
    args=script_args,
    exp_type=exp_type,
)
clean_gpu_memory()

online_dataset = None
for i in range(script_args.num_online_iterations):
    print('iteration {} ...'.format(i))
    checkpoint_path = os.path.join(save_path, 'model_iter{}'.format(i))
    if i == 0 and script_args.training_steps == 0:
        model_path = base_model_name
        peft_name = base_model_name
    else:
        model_path = checkpoint_path
        peft_name = checkpoint_path

    ### generation
    if script_args.num_generation_samples > 0:
        generate_data(
            model_path,
            reward_model_path_list=reward_model_path_list,
            tokenizer_name=tokenizer_name,
            rm_tokenizer_path_list=rm_tokenizer_path_list,
            dataset=dataset,
            save_path=os.path.join(save_path, 'model_iter{}'.format(i)),
            peft_name=peft_name,
            reward_stats_path=reward_stats_path,
            iter=i,
            args=script_args,
            exp_type=exp_type,
        )

    clean_gpu_memory()
    ### merging original data and generated data
    info_path = os.path.join(script_args.save_directory, script_args.wandb_name,'reward_info.npy')
    merged_data, online_dataset = merge_dataset(dataset, online_dataset, checkpoint_path, tokenizer_name, 
                                                info_path=info_path, exp_type=exp_type, quantile_threshold=script_args.quantile_threshold,
                                                sample_origin=script_args.num_origin_samples)
    train_model(
        base_model_name=model_path,
        peft_name=peft_name,
        reward_model_path_list=reward_model_path_list,
        train_dataset=merged_data,
        save_path=save_path + '/model_iter{}'.format(i+1),
        tokenizer_name=tokenizer_name,
        rm_tokenizer_path_list=rm_tokenizer_path_list,
        training_steps=script_args.online_training_steps,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type='constant',
        iter=i+1,
        args=script_args,
        exp_type=exp_type,
    )
    clean_gpu_memory()



