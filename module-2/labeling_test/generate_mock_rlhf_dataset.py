from transformers import pipeline, set_seed
import json
from typing import List 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



def generate_text(prompt: str, max_length: int, num_return_sequences: int, model_name: str):
    

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=num_return_sequences, num_beams=50)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return result


def generate_examples(prompt_list: List[str], model_name: str = 'google/flan-t5-large', max_length: int = 50, num_return_sequences: int = 2, seed: int = 42):
    
    set_seed(seed)
    examples = []
    for prompt in prompt_list:
        result = generate_text(prompt=prompt, max_length=max_length, num_return_sequences=num_return_sequences, model_name=model_name)
        
        example = {'prompt': prompt}
        for i, generated_text in enumerate(result):
            answer = generated_text.lstrip().strip()
            example[f'answer{i + 1}'] = answer
        examples.append(example)
        print(json.dumps(example, indent=2))
    return examples


def generate_dataset():
    prompts = [
        # "What is the latest news on the stock market?",
        # "What is the current state of the economy?",
        # "What are the latest developments in technology?",
        # "What is the political situation in the Middle East?",
        # "What are the latest trends in fashion and beauty?",
        # "What are the top travel destinations for this year?",
        # "What are some healthy recipes for a vegan diet?",
        # "What are the most important events happening in the world today?",
        # "What are some tips for improving mental health?",
        # "What are the best ways to save money for retirement?",
        # "What are some popular new books or movies?",
        # "What are some effective ways to reduce stress?",
        # "What are the latest developments in artificial intelligence?",
        # "What are some top-rated restaurants in your city?",
        # "What are the best ways to stay fit and healthy?",
        # "What are some tips for successful entrepreneurship?",
        # "What are some effective ways to improve productivity?",
        # "What are the latest developments in climate change research?",
        # "What are some top-rated TV shows or movies on streaming services?",
        # "What are some fun activities to do on weekends?",
        # "What are some effective ways to manage time and prioritize tasks?",
        # "What are the latest trends in home decor and design?",
        # "What are the best ways to develop a successful career?",
        # "What are some popular new products or gadgets?",
        # "What are some effective ways to improve communication skills?",
        # "What are some tips for successful relationships?",
        # "What are the latest developments in space exploration?",
        # "What are some top-rated online courses or certifications?",
        # "What are some effective ways to improve public speaking skills?",
        # "What are the latest trends in digital marketing?",
        "What are some fun and creative DIY projects?",
        "What are some effective ways to improve leadership skills?"
    ]

    generated_examples = generate_examples(prompts)
    with open('generated_examples.json', 'w') as f:
        json.dump(generated_examples, f)

if __name__ == '__main__':
    generate_dataset()