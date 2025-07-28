# scripts/download_and_convert_weights.py

import os
import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from safetensors import safe_open

def convert_weights_memory_efficient():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    print(f"Downloading model snapshot for '{model_id}'...")
    model_snapshot_path = snapshot_download(repo_id=model_id)
    print(f"Snapshot downloaded to: {model_snapshot_path}")

    output_dir = os.path.expanduser(f"~/flexgen_weights/{model_id}-np")
    os.makedirs(output_dir, exist_ok=True)

    safetensors_files = [f for f in os.listdir(model_snapshot_path) if f.endswith('.safetensors')]
    
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_snapshot_path}")

    print(f"Found {len(safetensors_files)} safetensors files. Converting to numpy...")

    for filename in tqdm(safetensors_files, desc="Converting shards"):
        filepath = os.path.join(model_snapshot_path, filename)
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            tensor_keys = f.keys()
            
            for key in tqdm(tensor_keys, desc=f"Converting tensors in {filename}", leave=False):
                param = f.get_tensor(key)
               
                if param.dtype == torch.bfloat16:
                    np_param = param.to(torch.float16).numpy()
                else:
                    np_param = param.numpy()
                
                param_path = os.path.join(output_dir, key)
                
                os.makedirs(os.path.dirname(param_path), exist_ok=True)
                
                np.save(param_path, np_param)

    print("Weight conversion complete.")

if __name__ == "__main__":
    convert_weights_memory_efficient()
