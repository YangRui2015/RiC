import os
from accelerate import Accelerator
import torch
from datasets import load_from_disk, disable_caching
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import numpy as np
import pandas as pd
from peft import LoraConfig, PeftModel
from utils import Instructions, load_main_tokenizer, save_configs, Instructions_summary_n, print_trainable_parameters
from trl import set_seed
disable_caching()


def train_model(
    base_model_name,
    reward_model_path_list,
    train_dataset,
    save_path,
    tokenizer_name=None,
    rm_tokenizer_path_list=None,
    peft_name=None,
    generated_dataset=None,
    training_steps=None,
    learning_rate=None,
    iter=0,
    lr_scheduler_type='linear',
    args=None,
    exp_type='assistant',
):
    set_seed(8888 + iter)
    print('base model: ', base_model_name)
    training_args = TrainingArguments(
            max_steps=training_steps,
            output_dir=os.path.join(args.save_directory, args.wandb_name),
            dataloader_drop_last=True,
            eval_steps=training_steps*2, # do not evaluate during training
            save_steps=training_steps*2,
            save_strategy='steps', 
            logging_steps=10,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=0,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=False,
            weight_decay=0.01,
            bf16=True,
            run_name=args.wandb_name,
            report_to='none',
            ddp_find_unused_parameters=False,
        )
    
    # # save training args
    save_configs(training_args, save_path)
    accelerator = Accelerator()
    process_id = Accelerator().local_process_index 
    gpu_id = process_id
    print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

    lora_config = LoraConfig(
        r=64, 
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    tokenizer = load_main_tokenizer(tokenizer_name)

    ### load dataset when input a path
    if type(train_dataset) == str:
        train_dataset = load_from_disk(train_dataset)

    selected_index = np.arange(0, len(train_dataset))
    np.random.shuffle(selected_index)
    dataset = train_dataset.select(selected_index)
    print(f"Size of the train set: {len(dataset)}")

    #### training 
    if training_steps > 0:
        if args.load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
                load_in_8bit=True, device_map=gpu_id)
        else: # load in bf 16
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
                torch_dtype=torch.bfloat16, device_map=gpu_id)

        model.resize_token_embeddings(len(tokenizer))
        if peft_name is not None:
            model = PeftModel.from_pretrained(model, peft_name, is_trainable=True)

        print_trainable_parameters(model)
        collator = DataCollatorForCompletionOnlyLM(
                        response_template=Instructions.response_split if exp_type == 'assistant' else Instructions_summary_n.response_split, 
                        tokenizer=tokenizer, mlm=False)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
            packing=False,
            dataset_text_field="query",
            data_collator=collator,
        )
        trainer.train()
        if process_id == 0:
            print("Saving last checkpoint of the model")
            trainer.model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
    
    # wait for the main process
    accelerator.wait_for_everyone()
    return train_dataset

