import argparse
import json
import os
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import string
import time
from collections import Counter

import sys
#~/specache-project/speedup/scripts/run_benchmark.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

SRC_PATH = os.path.join(PROJECT_ROOT, 'speedup', 'src')
sys.path.insert(0, SRC_PATH)


from flexgen.mistral_config import get_mistral_config
from flexgen.flex_mistral import MistralLM, Policy
from flexgen.compression import CompressionConfig
from flexgen.utils import ExecutionEnv
from flexgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice

def build_qasper_prompt(item, tokenizer, max_length):
    
    question = item['input']
    context = item['context']

    user_content = (
        "You are a helpful research assistant.\n\n"
        "Read the following context and answer the question concisely.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt 

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def postprocess_qasper_answer(pred):
    text = pred.strip()
    if text.lower().startswith("answer:"):
        text = text[len("answer:"):].strip()

    para_end = text.find("\n\n")
    if para_end != -1:
        text = text[:para_end].strip()
    return text

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def evaluate_qasper(predictions, references):
    f1 = 0.0
    for pred, ref_list in zip(predictions, references):
        f1 += max([f1_score(pred, ref) for ref in ref_list])
    return {"f1_score": f1 / len(predictions)}

TASK_MAPPING = {
    "qasper": {
      "prompt_builder": build_qasper_prompt,
      "postprocessor": postprocess_qasper_answer,
      "evaluator": evaluate_qasper,
      "ref_key": "answers"
    }
    #add another benchmark test.
}

def to_str_list(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        if len(x) > 0 and isinstance(x[0], dict):
            return [d.get('text', str(d)) for d in x]
        return [str(e) for e in x]
    if isinstance(x, dict):
        for k in ('text', 'answer', 'answers')
        if k in x and isinstance(x[k], str):
            return [x[k]]
        if k in x and isinstance(x[k], list):
            return [str(e) for e in x[k]]
        return [str(x)]
    return [str(x)]

def run_benchmark(config):
    
    model_args = config['model_args']
    policy_args = config['policy_args']
    benchmark_args = config['benchmark_args']

    task_name = benchmark_args['task_name']
    if task_name not in TASK_MAPPING:
        raise ValueError(f"Task '{task_name}' is not supported.")

    task_handler = TASK_MAPPING[task_name]

    print("Initializing model and environment...")
    tokenizer = AutoTokenizer.from_pretrained(model_args['model'], padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mistral_config = get_mistral_config(model_args['model'])

    gpu = TorchDevice("cuda:0", config=mistral_config)
    cpu = TorchDevice("cpu", config=mistral_config)
    disk = TorchDisk(model_args['offload_dir'])
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
   
    policy = Policy(
        gpu_batch_size=policy_args['gpu_batch_size'],
        num_gpu_batches=policy_args['num_gpu_batches'],
        w_gpu_percent=policy_args['percent'][0], w_cpu_percent=policy_args['percent'][1],
        cache_gpu_percent=policy_args['percent'][2], cache_cpu_percent=policy_args['percent'][3],
        act_gpu_percent=policy_args['percent'][4], act_cpu_percent=policy_args['percent'][5],
        overlap=False, pin_weight=policy_args['pin_weight'], cpu_cache_compute=False,
        compress_weight=False, comp_weight_config=None,
        compress_cache=False, comp_cache_config=None
    )

    model=MistralLM(mistral_config, env, model_args['path'], policy, model_id=model_args['model'])
    print("Model initialized.")

    print(f"Loading dataset for {task_name}...")
    dataset = load_dataset("THUDM/LongBench", task_name, split="test")

    #PRACTICAL_MAX_LEN = 2048
    #max_input_len = min(mistral_config.max_position_embeddings - benchmark_args['max_new_tokens'], PRACTICAL_MAX_LEN)
    
    max_input_len = mistral_config.max_position_embeddings

    predictions = []
    references = []

    total_batch_size = policy.gpu_batch_size * policy.num_gpu_batches
    total_duration_sec = 0.0
    total_prompt_tokens = 0
    total_generated_tokens = 0

    for i in tqdm(range(0, len(dataset), total_batch_size), desc="Running benchmark..."):
        batch_slice = dataset[i : i + total_batch_size]
        if len(batch_slice['input']) < total_batch_size:
            continue

        batch_prompts = []
        for j in range(total_batch_size):
            item = {key: batch_slice[key][j] for key in batch_slice.keys()}
            prompt = task_handler['prompt_builder'](item, tokenizer, max_input_len)
            batch_prompts.append(prompt)

        inputs_np = tokenizer(
            batch_prompts, 
            return_tensors="np", 
            padding=True,
            ).input_ids
        
        print(f'Input shape for batch {i//total_batch_size + 1}: {inputs_np.shape}')
        start_time = time.time()

        output_ids = model.generate(
            inputs=inputs_np,
            max_new_tokens=benchmark_args['max_new_tokens'],
        )
        
        end_time = time.time()
        total_duration_sec += (end_time - start_time)

        prompt_tokens_in_batch = np.sum(inputs_np != tokenizer.pad_token_id)
        generated_tokens_in_batch = np.sum(output_ids[:, inputs_np.shape[1]:] != tokenizer.pad_token_id)

        total_prompt_tokens += prompt_tokens_in_batch
        total_generated_tokens += generated_tokens_in_batch

        output_texts = tokenizer.batch_decode(output_ids[:, inputs_np.shape[1]:], skip_special_tokens=True)

        for idx, text in enumerate(output_texts):
            prediction = task_handler['postprocessor'](text)
            predictions.append(prediction)

            raw_ref = batch_slice[task_handler['ref_key']][idx]
            references.append(to_str_list(raw_ref))

    print("Evaluating results...")
    metrics = task_handler['evaluator'](predictions, references)

    total_tokens = total_prompt_tokens + total_generated_tokens
    overall_throughput = total_tokens / total_duration_sec if total_duration_sec > 0 else 0
    generation_throughput = total_generated_tokens / total_duration_sec if total_duration_sec > 0 else 0

    metrics['total_duration_sec'] = round(total_duration_sec, 2)
    metrics['total_prompt_tokens'] = int(total_prompt_tokens)
    metrics['total_generated_tokens'] = int(total_generated_tokens)
    metrics['overall_throughput_tokens_per_sec'] = round(overall_throughput, 2)
    metrics['generation_throughput_tokens_per_sec'] = round(generation_throughput, 2)

    print(f"\n---Result for {task_name}---")
    print(json.dumps(metrics,indent=2))

    os.makedirs(os.path.dirname(benchmark_args['output_file']), exist_ok=True)
    with open(benchmark_args['output_file'], 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {benchmark_args['output_file']}")

    env.close_copy_threads()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the benchmark config JSON file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    run_benchmark(config)
