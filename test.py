from models.modeling_llama import LlamaPreTrainedModel,LlamaForCausalLM
from models.configuration_llama import LlamaConfig
from transformers import LlamaTokenizer
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling,TrainerCallback,TrainerControl,TrainerState,HfArgumentParser
from datasets import load_from_disk, load_dataset
import argparse
import math
from itertools import chain
import os
from dataclasses import dataclass,field

# SEE https://huggingface.co/docs/transformers/main/main_classes/deepspeed#deepspeed-notebook
tokenizer_path = '/mnt/petrelfs/share_data/llm_llama2/llama-2-7b-hf/'
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
# tokenizer.model_max_length = 1024

def tokenize_function(examples):
    return tokenizer(examples['text'])

def group_texts(examples):
    max_seq_length=1024
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length // max_seq_length) * max_seq_length

    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result

class ProfilerCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    def __init__(self, prof):
        self.prof = prof

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.prof.step()
        print("Profiler Step Added")


@dataclass
class ProfileArguments:
    """
    Arguments relating to Dropping.
    """
    tensorboard_trace_handler: str = field(default="llama_trace", metadata={"help": "Name for the tensorboard trace."})



@dataclass
class DropTrainingArguments:
    """
    Arguments relating to Dropping.
    """
    drop_deep: bool = field(default=False, metadata={"help": "Random drop layers."})
    drop_weight: bool = field(default=False, metadata={"help": "Random drop K-V vectors."})

def main():
    parser = HfArgumentParser((ProfileArguments,DropTrainingArguments,TrainingArguments)) #argparse.ArgumentParser()
    # parser.add_argument("--drop_deep",action="store_true")
    # parser.add_argument("--drop_weight",action="store_true")
    # parser.add_argument("--important_sampling",action="store_true")
    # args = parser.parse_args()
    profile_args,drop_args,training_args = parser.parse_args_into_dataclasses()
    loss_fct = nn.CrossEntropyLoss()

    def preprocess_logits_for_metrics(logits, labels):
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, labels

    def compute_metrics(pred):
        loss = torch.from_numpy(pred.predictions[0])
        return {'perplexity': torch.exp(loss.mean())}

    # config = LlamaConfig(drop_deep=drop_args.drop_deep,drop_weight=drop_args.drop_weight)
    # model = LlamaForCausalLM(config)

    config = LlamaConfig(vocab_size=32000,num_attention_heads=32,hidden_size=2048,intermediate_size=5632,max_position_embeddings=2048,\
                        num_hidden_layers=22, drop_deep=drop_args.drop_deep,drop_weight=drop_args.drop_weight,\
                            _attn_implementation = "eager",use_cache=False)
    model = LlamaForCausalLM(config)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter number={count_parameters(model)} drop_deep={drop_args.drop_deep}")
    # model = GPT2LMHeadModel.from_pretrained())
    
    # model.drop_weight = False
    # model.drop_deep=False

    

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # raw_datasets = load_dataset("openwebtext",cache_dir="/mnt/lustre/share_data/wuzijian/cache/cache/huggingface/datasets")
    # raw_datasets = raw_datasets['train'].train_test_split(train_size=10000,test_size=10000,shuffle=False,load_from_cache_file=True, train_indices_cache_file_name = "/mnt/lustre/share_data/wuzijian/cache/cache_train_split.arrow",test_indices_cache_file_name="/mnt/lustre/share_data/wuzijian/cache/cache_process_split.arrow")
    # # raw_datasets = load_dataset("RedPajama-Data-1T-Sample/RedPajama-Data-1T-Sample.py",trust_remote_code=True)
    # tokenized_datasets = raw_datasets.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=64,
    #     remove_columns=raw_datasets["train"].column_names,
    #     desc="Running tokenizer on every text in dataset"
    #     ,cache_file_names={"train":"/mnt/lustre/share_data/wuzijian/cache_process_train0.arrow","test":"/mnt/lustre/share_data/wuzijian/cache_process_test0.arrow"},
    # )
    # tokenized_datasets.save_to_disk("./local_datasets0/")
    # random_prompts = torch.randint(tokenizer.vocab_size, (100000, 256), device=torch.cuda.current_device())
    # tokenized_datasets = load_from_disk("./local_datasets0/")
    # tokenized_datasets = tokenized_datasets.map(
    #     group_texts,
    #     batched=True,
    #     num_proc=64,
    #     desc=f"Grouping texts in chunks 1024",
    # )['train']

    # tokenized_datasets.save_to_disk("./local_datasets1_1024/")
    #   # random_prompts = torch.randint(tokenizer.vocab_size, (100000, 256), device=torch.cuda.current_device())
    tokenized_datasets = load_from_disk("./local_datasets1_1024/")
    # tokenized_datasets = load_dataset("togethercomputer/RedPajama-Data-1T-sample",trust_remote_code=True)

    print(tokenized_datasets)
    tokenized_datasets = tokenized_datasets.train_test_split(train_size=torch.distributed.get_world_size()*(5+6)*training_args.per_device_train_batch_size,test_size=2560)

    # training_args = TrainingArguments(output_dir='gpt2_output',
    #                                 run_name='gpt2_normal',
    #                                 per_device_train_batch_size = 4,
    #                                 per_device_eval_batch_size = 4,
    #                                 num_train_epochs=1,
    #                                 save_steps = 1000,
    #                                 evaluation_strategy='steps',
    #                                 eval_steps = 1000,
    #                                 logging_steps=5,
    #                                 learning_rate=6e-4,
    #                                 gradient_accumulation_steps=1,
    #                                 save_safetensors=True,
    #                                 fp16=True,
    #                                 fp16_full_eval=True,
    #                                 deepspeed='ds_config_single_GPU_no_offload.json')

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA], 
                                schedule=torch.profiler.schedule(skip_first=3, wait=0, warmup=2, active=6, repeat=1),
                                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_args.tensorboard_trace_handler),
                                profile_memory=True,
                                with_stack=True,
                                record_shapes=True) as prof:
    
        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        trainer.add_callback(ProfilerCallback(prof=prof))
        trainer.train()
        
        
        def filter_table(table_string):
            lines = table_string.split('\n')
            new_lines = []

            # Add the first three lines
            new_lines.extend(lines[:3])

            # Add lines that contain "#MODEL_"
            for line in lines[3:]:
                if "#MODEL_" in line or "ProfilerStep" in line \
                    or "deepspeed/runtime/engine.py(1912): backward" in line \
                    or "deepspeed/runtime/engine.py(2111): step" in line:
                    new_lines.append(line)

            return '\n'.join(new_lines)
        
        print(filter_table(prof.key_averages().table(sort_by="cpu_time_total", row_limit=999999)))
        # print("==================================\n")
        # print(prof.key_averages(group_by_stack_n=100).table(sort_by="self_cpu_time_total", row_limit=100))

if __name__ == '__main__':
    main()