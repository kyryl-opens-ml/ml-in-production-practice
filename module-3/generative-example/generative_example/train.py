from random import randrange
import torch
from datasets import load_dataset
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

def create_message_column(row):
    messages = []
    
    # Create a 'user' message dictionary with 'content' and 'role' keys.
    user = {
        "content": f"{row['context']}\n Input: {row['question']}",
        "role": "user"
    }
    
    # Append the 'user' message to the 'messages' list.
    messages.append(user)
    
    # Create an 'assistant' message dictionary with 'content' and 'role' keys.
    assistant = {
        "content": f"{row['answer']}",
        "role": "assistant"
    }
    
    # Append the 'assistant' message to the 'messages' list.
    messages.append(assistant)
    
    # Return a dictionary with a 'messages' key and the 'messages' list as its value.
    return {"messages": messages}

# 'format_dataset_chatml' is a function that takes a row from the dataset and returns a dictionary 
# with a 'text' key and a string of formatted chat messages as its value.
def format_dataset_chatml(row):
    # 'tokenizer.apply_chat_template' is a method that formats a list of chat messages into a single string.
    # 'add_generation_prompt' is set to False to not add a generation prompt at the end of the string.
    # 'tokenize' is set to False to return a string instead of a list of tokens.
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}


def train():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    dataset_name = "b-mc2/sql-create-context"
    dataset_split= "train"
    new_model = "text2sql"
    device_map = {"": 0}
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
    set_seed(1234)    


    dataset = load_dataset(dataset_name, split=dataset_split)
    print(f"dataset size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])


    print(dataset[randrange(len(dataset))])

    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = 'right'

    dataset_chatml = dataset.map(create_message_column)
    dataset_chatml = dataset_chatml.map(format_dataset_chatml)

    print(dataset_chatml[0])


    dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)

    subsample = True
    if subsample:
        dataset_chatml['train'] = dataset_chatml['train'].shuffle(seed=1234).select(range(7_000))
        dataset_chatml['test'] = dataset_chatml['test'].shuffle(seed=1234).select(range(700))

    dataset_chatml    


    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
        # If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'

        # This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
        print(attn_implementation)
    

    model_name = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
              model_id, torch_dtype=compute_dtype, trust_remote_code=True, device_map=device_map,
              attn_implementation=attn_implementation
    )

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
         

    import wandb

    project_name = "Phi3-mini-ft-python-code"
    wandb.init(project=project_name, name = "ml-in-production-practice")


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
     