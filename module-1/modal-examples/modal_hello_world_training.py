import modal

app = modal.App("function-calling-finetune")
image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "transformers==4.51.2",
            "peft==0.15.1",
            "bitsandbytes==0.45.4",
            "trl==0.16.1",
            "datasets==3.5.0",
            "torch==2.2.1",
            "accelerate==1.5.2",
            "wandb==0.19.8",
        ]
    )
    .env({"WANDB_PROJECT": "function-calling-finetune"})
)

with image.imports():
    from enum import Enum
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, TaskType, PeftConfig, PeftModel


DATASET_NAME = "Jofthomas/hermes-function-calling-thinking-V1"
USERNAME = "truskovskiyk"
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = "gemma-3-4b-it-function-calling"


@app.function(
    image=image,
    cloud="aws",
    gpu="H200",
    timeout=86400,
    secrets=[modal.Secret.from_name("training-config")],
)
def function_calling_finetune():
    set_seed(42)

    dataset_name = DATASET_NAME
    username = USERNAME
    model_name = MODEL_NAME
    output_dir = OUTPUT_DIR

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

    def preprocess(sample):
        messages = sample["messages"]
        first_message = messages[0]
        if first_message["role"] == "system":
            system_message_content = first_message["content"]
            messages[1]["content"] = (
                system_message_content
                + "Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n"
                + messages[1]["content"]
            )
            messages.pop(0)
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = load_dataset(dataset_name)
    dataset = dataset.rename_column("conversations", "messages")
    dataset = dataset.map(preprocess, remove_columns="messages")
    dataset = dataset["train"].train_test_split(0.1)

    sample = dataset["train"].select(range(1))
    print(f"Sample: {sample['text']}")

    class ChatmlSpecialTokens(str, Enum):
        tools = "<tools>"
        eotools = "</tools>"
        think = "<think>"
        eothink = "</think>"
        tool_call = "<tool_call>"
        eotool_call = "</tool_call>"
        tool_response = "<tool_reponse>"
        eotool_response = "</tool_reponse>"
        pad_token = "<pad>"
        eos_token = "<eos>"

        @classmethod
        def list(cls):
            return [c.value for c in cls]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list(),
    )
    tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(torch.bfloat16)

    rank_dimension = 16
    lora_alpha = 64
    lora_dropout = 0.05
    peft_config = LoraConfig(
        r=rank_dimension,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "gate_proj",
            "q_proj",
            "lm_head",
            "o_proj",
            "k_proj",
            "embed_tokens",
            "down_proj",
            "up_proj",
            "v_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )

    per_device_train_batch_size = 16
    per_device_eval_batch_size = 16
    gradient_accumulation_steps = 1
    logging_steps = 5
    learning_rate = 1e-4
    max_grad_norm = 1.0
    num_train_epochs = 3.0
    warmup_ratio = 0.1
    lr_scheduler_type = "cosine"
    max_seq_length = 1500

    training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="no",
        eval_strategy="epoch",
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        weight_decay=0.1,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
        bf16=True,
        hub_private_repo=False,
        push_to_hub=False,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=True,
        max_seq_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.config.use_cache = False
    trainer.model.generation_config.use_cache = False
    trainer.train()
    trainer.save_model()
    trainer.push_to_hub(f"{username}/{output_dir}")
    tokenizer.eos_token = "<eos>"
    tokenizer.save_pretrained(f"{username}/{output_dir}")
    tokenizer.push_to_hub(f"{username}/{output_dir}", token=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    secrets=[modal.Secret.from_name("training-config")],
)
def function_calling_inference():
    username = USERNAME
    output_dir = OUTPUT_DIR

    peft_model_id = f"{username}/{output_dir}"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.eval()

    prompt = """<bos><start_of_turn>human
    You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}, {'type': 'function', 'function': {'name': 'calculate_distance', 'description': 'Calculate the distance between two locations', 'parameters': {'type': 'object', 'properties': {'start_location': {'type': 'string', 'description': 'The starting location'}, 'end_location': {'type': 'string', 'description': 'The ending location'}}, 'required': ['start_location', 'end_location']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
    <tool_call>
    {tool_call}
    </tool_call>Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>

    Hi, I need to convert 500 USD to Euros. Can you help me with that?<end_of_turn><eos>
    <start_of_turn>model
    <think>"""

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_p=0.95,
        temperature=0.01,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(outputs[0]))
