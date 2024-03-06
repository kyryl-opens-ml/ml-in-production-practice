import torch
import typer

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import argparse
import torch
import os
import pandas as pd
import pickle
import warnings
from tqdm import tqdm

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# from lora_training.datasets_prep import get_newsgroup_data_for_ft

warnings.filterwarnings("ignore")






def training_llm(
    # pretrained_ckpt: str = "mistralai/Mistral-7B-v0.1", 

    # pretrained_ckpt: str = "microsoft/phi-1_5", 
    # pretrained_ckpt: str = "facebook/opt-350m", 
    
    lora_r: int = 8, 
    epochs: int = 5, 
    dropout: float = 0.1, 
    train_sample_fraction: float = 0.25
    ):
    
    
    train_dataset, test_dataset = get_newsgroup_data_for_ft(mode="train", train_sample_fraction=train_sample_fraction)
    print(f"Sample fraction:{train_sample_fraction}")
    print(f"Training samples:{train_dataset.shape}")
    print(f"Training sample idx = 0\n:{train_dataset['instructions'][0]}")

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    results_dir = f"experiments/classification"

    training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=100,
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
    )
    max_seq_length = 512  # max sequence length for model and packing of the dataset
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=training_args,
        dataset_text_field="instructions",
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    print("Experiment over")

def inference_llm(experiment_dir: str = 'experiments/classification'):
    _, test_dataset = get_newsgroup_data_for_ft(mode="inference")

    experiment = experiment_dir
    peft_model_id = f"{experiment}/assets"

    # load base LLM model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    results = []
    oom_examples = []
    instructions, labels = test_dataset["instructions"], test_dataset["labels"]

    for instruct, label in tqdm(zip(instructions, labels)):
        input_ids = tokenizer(
            instruct, return_tensors="pt", truncation=True
        ).input_ids.cuda()

        with torch.inference_mode():
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,
                    do_sample=True,
                    top_p=0.95,
                    temperature=1e-3,
                )
                result = tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True
                )[0]

                result = result[len(instruct) :]
                print(result)
            except:
                result = ""
                oom_examples.append(input_ids.shape[-1])

            results.append(result)

    metrics = {
        "micro_f1": f1_score(labels, results, average="micro"),
        "macro_f1": f1_score(labels, results, average="macro"),
        "precision": precision_score(labels, results, average="micro"),
        "recall": recall_score(labels, results, average="micro"),
        "accuracy": accuracy_score(labels, results),
        "oom_examples": oom_examples,
    }
    print(metrics)
    print(f"Completed experiment {peft_model_id}")
    print("----------------------------------------")


def cli():
    app = typer.Typer()
    app.command()(training_llm)
    app.command()(inference_llm)
    app()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--experiment_dir",
#         default="experiments/classification-sampleFraction-0.1_epochs-5_rank-8_dropout-0.1",
#     )

#     args = parser.parse_args()
#     main(args)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pretrained_ckpt", default="mistralai/Mistral-7B-v0.1")
#     parser.add_argument("--lora_r", default=8, type=int)
#     parser.add_argument("--epochs", default=5, type=int)
#     parser.add_argument("--dropout", default=0.1, type=float)
#     parser.add_argument("--train_sample_fraction", default=0.99, type=float)

#     args = parser.parse_args()
#     main(args)

if __name__ == '__main__':
    cli()