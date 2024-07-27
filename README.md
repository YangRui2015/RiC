# RiC: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment

Code for the ICML 2024 paper "Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment". This repo is based on [trl](https://github.com/huggingface/trl).


## 1. Installation
Install the requirements.
```
pip install -r requirements.txt
```

The following issue has been fixed using token_ids directly for response_template as in https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completions-only.
```
Fix the trl package.


The trl package currently has an issue with its 'DataCollatorForCompletionOnlyLM' implementation, wherein the first token of the template may be nonsensical and unmatched. Hence, it is necessary to make modifications to the trl/trainer/utils.py file within your trl package (usually under the path {path to conda or python}/lib/python3/site-packages), specifically the 'torch_call' function starting from line 119. Upon installing the trl package based on the requirements.txt, you can easily replace the trl/trainer/utils.py with the utils.py file located in the 'fix_trl' directory.
```

## 2. Usage
### 2.1 RiC
```
Note: To reproduce the results in the paper, ensure the same training steps, for example, (4 processes, batchsize=2, steps=20000), and load the model in 8bit (--load_in_8bit True). We have observed notable performance differences between 8-bit and 16-bit models, particularly in the summary task.
```


* **Preparing datasets**: the first step is to generate dataset for RiC training.
```
cd ./ric
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch prepare_dataset_with_rewards.py --reward_names 'harmless,helpful' --exp_type 'assistant' --save_directory './datasets/train_harmhelp.hf' 
```
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch prepare_dataset_with_rewards.py --reward_names 'summary,faithful' --exp_type 'summary' --save_directory './datasets/summary_pref1faithful.hf'
```
Here, 'exp_type' includes two types of tasks, i.e., 'assistant' and 'summary'. Moreover, 'reward_names' can be sampled from \{'harmless', 'helpful', 'humor', 'summary', 'deberta', 'faithful'\}, where the first three are used for the 'assistant' task while the last three are used for the 'summary' task.

* **Training**: you can train RiC through the main.py file.

Offline training:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --train_dataset_path {path_to_corresponding_dataset} --exp_type 'assistant' --reward_names 'harmless,helpful' --training_steps 20000 --num_online_iterations 0 --wandb_name 'ric_assistant_harmlesshelpful_offline20000' --batch_size 2 --load_in_8bit True
```
Offline training + online training:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py --train_dataset_path {path_to_corresponding_dataset} --exp_type 'assistant' --reward_names 'harmless,helpful' --training_steps 20000 --online_training_steps 4000 --num_online_iterations 2 --wandb_name 'ric_assistant_harmlesshelpful_offline20000_onlineiter2' --batch_size 2 --load_in_8bit True
```

* **Evaluation**: After training, models are saved into 'save_directory' path. You can evaluate models using the evaluation.py.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch evaluation.py --peft_name {path_to_the_saved_peft} --reward_names 'harmless,helpful' --exp_type 'assistant' --wandb_name {path_of_the_experiment}
```

### 2.2 SFT
* Training a SFT model:
```
cd ./sft
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch sft.py --base_model_name 'meta-llama/Llama-2-7b-hf' --exp_type 'summary' --wandb_name {name_of_the_experiment} 
```
* Evaluating a SFT model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch eval_multiobjective.py --base_model_name {path_to_the_saved_SFT_model} --exp_type 'summary' --wandb_name {name_of_the_experiment} --reward_names 'summary,faithful'
```

* Merging the SFT model if using lora for finetuning:
```
python3 merge_sft_lora.py --base_model_name 'meta-llama/Llama-2-7b-hf' --lora_name {path_to_the_lora_file} --save_directory {path_to_the_saving_directory}
```

### 2.3 MORLHF 
MORLHF optimizes preference weighted rewards using PPO, based on the SFT model in 2.2.

Train MORLHF:
```
cd ./ppo
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch morlhf.py --preference 0.5 --base_model_name {path_to_the_sft_model} --reward_names 'harmless,helpful' --exp_type 'assistant' --wandb_name 'morlhf_llamma2'
```
Here, the 'preference' is the preference for the first reward.

Evaluate MORLHF model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch eval_ppo_single_model.py --base_model_name {path_to_the_saved_morlhf_model} --reward_names 'harmless,helpful' --exp_type 'assistant' --wandb_name 'eval_morlhf_llamma2'
```

### 2.4 Rewarded Soups
Rewarded Soups train $N$ PPO models for $N$ reward models.  

Train PPO model for each reward model:
```
cd ./ppo
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ppo.py --reward_name 'harmless' --base_model_name {path_to_the_SFT_model} --exp_type 'assistant' --wandb_name {name_for_this_experiment}
```
Evaluate Rewarded Soups with multiple PPO models:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch eval_rewarded_soups.py --reward_names 'harmless,helpful' --base_model_path1 {path_to_the_PPO_model_for_harmless} --base_model_path2 {path_to_the_PPO_model_for_helpful} --exp_type 'assistant' --wandb_name 'eval_pposoups_harmless_helpful'
```


## 3. Citation
If you find our work useful for your research, please cite:
```
@article{yang2024rewards,
  title={Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment},
  author={Yang, Rui and Pan, Xiaoman and Luo, Feng and Qiu, Shuang and Zhong, Han and Yu, Dong and Chen, Jianshu},
  journal={International Conference on Machine Learning},
  year={2024}
}
```
