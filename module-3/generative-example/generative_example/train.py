from random import randrange
import torch
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline,
    HfArgumentParser
)
from pathlib import Path
import logging
from trl import SFTTrainer
from functools import partial
from datasets import load_metric

from generative_example.config import DataTrainingArguments, ModelArguments
from generative_example.utils import setup_logger


logger = logging.getLogger(__name__)


def create_message_column(row):
    messages = []
    user = {
        "content": f"{row['context']}\n Input: {row['question']}",
        "role": "user"
    }
    messages.append(user)
    assistant = {
        "content": f"{row['answer']}",
        "role": "assistant"
    }
    messages.append(assistant)
    return {"messages": messages}

def format_dataset_chatml(row, tokenizer):
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}


def process_dataset(model_id: str, train_file: str, test_file: str) -> DatasetDict:
    
    dataset = DatasetDict({
        'train': Dataset.from_json(train_file),
        'test': Dataset.from_json(test_file),
    })
    

    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = 'right'

    dataset_chatml = dataset.map(create_message_column)
    dataset_chatml = dataset_chatml.map(partial(format_dataset_chatml, tokenizer=tokenizer))
    return dataset_chatml


def get_model(model_id: str, device_map):
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
        # If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'

        # This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
        print(attn_implementation)
    

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
              model_id, torch_dtype=compute_dtype, trust_remote_code=True, device_map=device_map,
              attn_implementation=attn_implementation
    )
    return tokenizer, model

def get_config(config_path: Path):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(config_path)
    return model_args, data_args, training_args

def train(config_path: Path):
    setup_logger(logger)

    model_args, data_args, training_args = get_config(config_path=config_path)
    
    logger.info(f"model_args = {model_args}")
    logger.info(f"data_args = {data_args}")
    logger.info(f"training_args = {training_args}")


    device_map = {"": 0}
    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]

    set_seed(training_args.seed)    

    dataset_chatml = process_dataset(model_id=model_args.model_id, train_file=data_args.train_file, test_file=data_args.test_file)
    logger.info(dataset_chatml['train'][0])

    tokenizer, model = get_model(model_id=model_args.model_id, device_map=device_map)
    peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
    )
         
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_chatml['train'],
        eval_dataset=dataset_chatml['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model()
     





    # def test_inference(prompt):
    #     prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    #     outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95,
    #                max_time= 180) #, eos_token_id=eos_token)
    #     return outputs[0]['generated_text'][len(prompt):].strip()

    # def calculate_rogue(row):
    #     response = test_inference(row['messages'][0]['content'])
    #     result = rouge_metric.compute(predictions=[response], references=[row['output']], use_stemmer=True)
    #     result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    #     result['response']=response
    #     return result

    # rouge_metric = load_metric("rouge", trust_remote_code=True)
    # metricas = dataset_chatml['test'].select(range(0,500)).map(calculate_rogue, batched=False)

    # print("Rouge 1 Mean: ",np.mean(metricas['rouge1']))
    # print("Rouge 2 Mean: ",np.mean(metricas['rouge2']))
    # print("Rouge L Mean: ",np.mean(metricas['rougeL']))
    # print("Rouge Lsum Mean: ",np.mean(metricas['rougeLsum']))