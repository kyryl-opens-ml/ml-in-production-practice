from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from trl import SFTTrainer, is_xpu_available
from accelerate import Accelerator
import torch

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")


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



instruction_template = "### Human:"
response_template = "### Assistant:"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)


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
    dataset_text_field="text",
    data_collator=collator,
    peft_config=peft_config
)

trainer.train()

trainer.save_model(output_dir)
