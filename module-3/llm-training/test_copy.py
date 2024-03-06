from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from trl import SFTTrainer, is_xpu_available
from accelerate import Accelerator
import torch

dataset = load_dataset("imdb", split="train")

quantization_config = BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=True)
device_map = ({"": f"xpu:{Accelerator().local_process_index}"} if is_xpu_available() else {"": Accelerator().local_process_index})
torch_dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")


model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m",
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
)



# def formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['instruction'])):
#         text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
#         output_texts.append(text)
#     return output_texts

def formatting_func(example):
    text = f"### Review: {example['text']}\n ### Answer: {'Positive' if example['label'] == 1 else 'Negative'}"
    return text

response_template = " ### Answer:"
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    bias="none",
    task_type="CAUSAL_LM",
)
output_dir = 'training'

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    # learning_rate=script_args.learning_rate,
    # logging_steps=script_args.logging_steps,
    # num_train_epochs=script_args.num_train_epochs,
    # max_steps=script_args.max_steps,
    # report_to=script_args.log_with,
    # save_steps=script_args.save_steps,
    # save_total_limit=script_args.save_total_limit,
    # push_to_hub=script_args.push_to_hub,
    # hub_model_id=script_args.hub_model_id,
    # gradient_checkpointing=script_args.gradient_checkpointing,
    # TODO: uncomment that on the next release
    # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
)

trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    packing=True,
    formatting_func=formatting_func,
    peft_config=peft_config
)

trainer.train()

trainer.save_model(output_dir)




from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from trl import SFTTrainer, is_xpu_available
from accelerate import Accelerator
import torch


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("training/checkpoint-1000/")
dataset = load_dataset("imdb", split="train")
example = dataset[0]
inference(dataset[15000])


def inference(example):
    
    text = f"### Review: {example['text']}\n ### Answer:"

    input_ids = tokenizer(text, return_tensors="pt", truncation=True).input_ids

    outputs = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=True, top_p=0.95, temperature=1e-3,)
    result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    print(result)
