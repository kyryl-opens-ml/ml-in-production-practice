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
    pipeline
)
from trl import SFTTrainer
from functools import partial
from datasets import load_metric

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


def process_dataset(model_id: str) -> DatasetDict:
    
    dataset = DatasetDict({
        'train': Dataset.from_json('data/train.json'),
        'test': Dataset.from_json('data/test.json'),
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

def train():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    device_map = {"": 0}
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
    set_seed(1234)    

    dataset_chatml = process_dataset(model_id=model_id)
    print(dataset_chatml['train'][0])

    tokenizer, model = get_model(model_id=model_id, device_map=device_map)
    # if torch.cuda.is_bf16_supported():
    #     compute_dtype = torch.bfloat16
    #     attn_implementation = 'flash_attention_2'
    #     # If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
    # else:
    #     compute_dtype = torch.float16
    #     attn_implementation = 'sdpa'

    #     # This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
    #     print(attn_implementation)
    

    # model_name = "microsoft/Phi-3-mini-4k-instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)
    # tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    # tokenizer.padding_side = 'left'

    # model = AutoModelForCausalLM.from_pretrained(
    #           model_id, torch_dtype=compute_dtype, trust_remote_code=True, device_map=device_map,
    #           attn_implementation=attn_implementation
    # )

    args = TrainingArguments(
            output_dir="./phi-3-mini-LoRA",
            evaluation_strategy="steps",
            do_eval=True,
            optim="adamw_torch",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=8,
            log_level="debug",
            save_strategy="epoch",
            logging_steps=100,
            learning_rate=1e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            eval_steps=100,
            num_train_epochs=3,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            report_to="wandb",
            seed=42,
    )

    peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
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
        args=args,
    )

    trainer.train()
    trainer.save_model()
     





    def test_inference(prompt):
        prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95,
                   max_time= 180) #, eos_token_id=eos_token)
        return outputs[0]['generated_text'][len(prompt):].strip()

    def calculate_rogue(row):
        response = test_inference(row['messages'][0]['content'])
        result = rouge_metric.compute(predictions=[response], references=[row['output']], use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result['response']=response
        return result

    rouge_metric = load_metric("rouge", trust_remote_code=True)
    metricas = dataset_chatml['test'].select(range(0,500)).map(calculate_rogue, batched=False)

    print("Rouge 1 Mean: ",np.mean(metricas['rouge1']))
    print("Rouge 2 Mean: ",np.mean(metricas['rouge2']))
    print("Rouge L Mean: ",np.mean(metricas['rougeL']))
    print("Rouge Lsum Mean: ",np.mean(metricas['rougeLsum']))