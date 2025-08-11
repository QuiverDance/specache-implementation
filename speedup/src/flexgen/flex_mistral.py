# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 snu-comparch contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
# -----------------------------------------------------------------------------------
# Modified by QuiverDance in 2025 : convert OPT into Mistral
# © 2025 QuiverDacne

import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.mistral_config import MistralConfig, get_mistral_config
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexgen.timer import timers
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"

@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    overlap: bool
    pin_weight: bool
    cpu_cache_compute: bool
    compress_weight: bool
    comp_weight_config: CompressionConfig
    compress_cache: bool
    comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        pin_memory = policy.pin_weight
        compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)
            
            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
        ret.append(weight)
    return ret


class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.hidden_size, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            ((v, h), dtype, path + "model.embed_tokens.weight.npy"),
            # Positional embedding is removed for Mistral (uses RoPE instead)
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        # Only load token embedding
        w_token = weight_home.val[0]
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst)[0],
            ))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Simplified forward pass for input embedding without pos_embed
        h = hidden.val
        
        if k == self.policy.num_gpu_batches - 1:
            (w_token,) = weight_read_buf.pop()
        else:
            (w_token,) = weight_read_buf.val
        
        # The actual computation is simplified here. The backend will handle it.
        # The RoPE is applied in the attention layer, not here.
        h = self.compute.mistral_input_embed(h, w_token)
        hidden.val = h
    
    def init_cache_one_gpu_batch(self, cache_home):
        pass

    def load_cache(self, cache_home, cache_read_buf, i):
        pass

    def store_cache(self, cache_home, cache_write_buf, i):
        pass

class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.hidden_size, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # Mistral's final RMSNorm and lm_head names
            ((h,), dtype, path + "model.norm.weight.npy"),
            ((v, h), dtype, path + "lm_head.weight.npy"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_lm_head = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_ln.smart_copy(dst2)[0],
                w_lm_head.smart_copy(dst1)[0]
            ))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        h = hidden.val
        if k == self.policy.num_gpu_batches - 1:
            w_ln, w_lm_head = weight_read_buf.pop()
        else:
            w_ln, w_lm_head = weight_read_buf.val
        
        h = self.compute.mistral_output_embed(h, w_ln, w_lm_head,
            self.task.do_sample, self.task.temperature, self.config.rms_norm_eps)
        hidden.val = h

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

class SelfAttention:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        
        # This determines where the attention calculation will happen (GPU or CPU).
        # For our baseline and SpeCache, we will always compute on GPU.
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, self.config.dtype)
        # CHANGED: Path and weight names for Mistral's attention layer
        path = os.path.join(os.path.join(path, f"model.layers.{self.layer_id}."))
        weight_specs = [
            ((h,), dtype, path + "input_layernorm.weight.npy"),
            ((h, h), dtype, path + "self_attn.q_proj.weight.npy"),
            ((self.config.num_key_value_heads * (h // self.config.num_attention_heads), h), dtype, path + "self_attn.k_proj.weight.npy"),
            ((self.config.num_key_value_heads * (h // self.config.num_attention_heads), h), dtype, path + "self_attn.v_proj.weight.npy"),
            ((h, h), dtype, path + "self_attn.o_proj.weight.npy"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_q, w_k, w_v, w_out = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            return (w_ln.smart_copy(dst2)[0],
                    w_q.smart_copy(dst1)[0],
                    w_k.smart_copy(dst1)[0],
                    w_v.smart_copy(dst1)[0],
                    w_out.smart_copy(dst1)[0])

    # init_cache, load_cache, store_cache, forward methods are kept as they are.
    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)

    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return
        
        k_home, v_home = cache_home.val
        
        dst = self.attention_compute
        
        #dst = self.compute

        # Our cache shape: (batch, num_heads, seq_len, head_dim)
        # We need to load the cache up to the current sequence length.
        # The sequence length for decoding step `i` is `prompt_len + i`.
        # the past length must be prompt_len + i - 1
        current_seq_len = self.task.prompt_len + i - 1

        # Remove the unnecessary `attn_sparsity` check and related logic.
        # We will always use dense attention for our baseline.
        # The indices define which part of the home cache to load.
        indices = (
            slice(0, k_home.shape[0]),       # Full batch
            slice(0, k_home.shape[1]),       # Full num_heads
            slice(0, current_seq_len),       # Up to the current token
            slice(0, k_home.shape[3]),       # Full head_dim
        )

        # Load both K and V caches into the read buffer for the compute device.
        cache_read_buf.store((
            k_home.smart_copy(dst, indices),
            v_home.smart_copy(dst, indices),
        ))

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()
        
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return
                
        if i == 0:  # Prefill stage
            indices = (
                slice(0, k_new.shape[0]),       # Full batch dimension
                slice(0, k_new.shape[1]),       # Full num_heads dimension
                slice(0, k_new.shape[2]),       # Slice for sequence length
                slice(0, k_new.shape[3]),       # Full head_dim dimension
            )
        else:  # Decoding stage
            # We need to update a single token position in the cache.
            # The new cache (k_new) will have seq_len = 1.
            pos = self.task.prompt_len + i - 1
            indices = (
                slice(0, k_new.shape[0]),       # Full batch dimension
                slice(0, k_new.shape[1]),       # Full num_heads dimension
                slice(pos, pos + 1),            # Slice for the single new token
                slice(0, k_new.shape[3]),       # Full head_dim dimension
            )
        
        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.dtype
    
    # The main logic changes for RoPE and GQA will be in the backend.
    # ... (rest of the SelfAttention class is conceptually similar) ...
    # We will just update the forward call signature
    def forward(self, hidden, cache_read_buf, attn_weights_tuple, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 8
        h, donate[0] = hidden.val, False

        w_ln, w_q, w_k, w_v, w_out = attn_weights_tuple
        
        if i == 0:  # prefill
            
            compute_device = self.attention_compute # (self.env.cpu or self.env.gpu)

            # Copy all required tensors to compute device (CPU or GPU)
            h_compute = hidden.val.copy(compute_device)
            mask_compute = attention_mask.val.copy(compute_device)
            w_ln_compute = w_ln.copy(compute_device)
            w_q_compute = w_q.copy(compute_device)
            w_k_compute = w_k.copy(compute_device)
            w_v_compute = w_v.copy(compute_device)
            w_out_compute = w_out.copy(compute_device)

            # Perform the MHA computation on the selected device
            # The `mha` function in the backend is device-agnostic.
            h_result, new_k_cache, new_v_cache = compute_device.mha(
                h_compute, mask_compute, 
                w_ln_compute, w_q_compute, w_k_compute, w_v_compute, w_out_compute,
                self.config, [False]*8, self.policy.compress_cache, self.policy.comp_cache_config
            )

            # Move results back to their designated homes
            # The hidden state must return to the primary compute device (GPU) for the next layer.
            hidden.val = h_result.move(self.compute)
            
            # The KV cache's home is determined by the offloading policy.
            cache_home_device = self.env.cpu
            cache_write_buf.store((
                new_k_cache.move(cache_home_device),
                new_v_cache.move(cache_home_device)
            ))
        else:  # decoding
            # Current absolute position where the new token's KV should be stored
            cache_pos = self.task.prompt_len + i  # padded_len 기반 (run_benchmark에서 그렇게 셋업됨)

            # Move inputs and weights to the compute device
            compute_device = self.attention_compute
            h_compute = hidden.val.copy(compute_device)

            w_ln_compute = attn_weights_tuple[0].copy(compute_device)
            w_q_compute  = attn_weights_tuple[1].copy(compute_device)
            w_k_compute  = attn_weights_tuple[2].copy(compute_device)
            w_v_compute  = attn_weights_tuple[3].copy(compute_device)
            w_out_compute= attn_weights_tuple[4].copy(compute_device)

            # Read previously-filled cache up to current position
            (k_cache_tuple, v_cache_tuple) = cache_read_buf.pop()
            k_cache = k_cache_tuple[0]
            v_cache = v_cache_tuple[0]

            # Compute attention for a single step and get ONLY the new KV slice
            h_result, k_new, v_new = compute_device.mha_gen(
                h_compute, w_ln_compute, w_q_compute, w_k_compute, w_v_compute, w_out_compute,
                self.config, k_cache, v_cache, [False]*2, self.policy.compress_cache, self.policy.comp_cache_config,
                cache_pos=cache_pos
            )

            hidden.val = h_result.move(self.compute)
            cache_write_buf.store((k_new.move(self.env.cpu), v_new.move(self.env.cpu)))


class MLP:
    # This class is heavily modified for Mistral's SwiGLU MLP.
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, f, dtype = (self.config.hidden_size, self.config.intermediate_size, self.config.dtype)
        path = os.path.join(os.path.join(path, f"model.layers.{self.layer_id}."))
        weight_specs = [
            # Weights for SwiGLU
            ((h,), dtype, path + "post_attention_layernorm.weight.npy"),
            ((f, h), dtype, path + "mlp.gate_proj.weight.npy"),
            ((f, h), dtype, path + "mlp.up_proj.weight.npy"),
            ((h, f), dtype, path + "mlp.down_proj.weight.npy"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_gate, w_up, w_down = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            # Do not store here. Return the loaded tensors instead.
            return (w_ln.smart_copy(dst2)[0],
                    w_gate.smart_copy(dst1)[0],
                    w_up.smart_copy(dst1)[0],
                    w_down.smart_copy(dst1)[0])

    def forward(self, hidden, cache_read_buf, mlp_weights_tuple, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 5
        h, donate[0] = hidden.val, False #for safety

        # 1. Determine the compute device based on the policy.
        #    Decoding (i > 0) is always on GPU.
        if i == 0 and self.policy.cpu_cache_compute:
            compute_device = self.env.cpu
        else:
            compute_device = self.env.gpu

        # 2. Unpack weights
        w_ln, w_gate, w_up, w_down = mlp_weights_tuple

        # 3. Move all required tensors to the compute device (CPU or GPU)
        h_compute = hidden.val.copy(compute_device)
        w_ln_compute = w_ln.copy(compute_device)
        w_gate_compute = w_gate.copy(compute_device)
        w_up_compute = w_up.copy(compute_device)
        w_down_compute = w_down.copy(compute_device)

        # 4. Perform the MLP computation on the selected device.
        mlp_result = compute_device.mistral_mlp(
            h_compute, 
            w_ln_compute, w_gate_compute, w_up_compute, w_down_compute,
            self.config.rms_norm_eps, [False]*5
        )
        
        # 5. Move the final result back to the primary compute device (GPU) and
        #    update the `hidden` ValueHolder for the next layer or output.
        hidden.val = mlp_result.move(self.compute)

    def init_cache_one_gpu_batch(self, cache_home):
        pass
    def load_cache(self, cache_home, cache_read_buf, i):
        pass
    def store_cache(self, cache_home, cache_write_buf, i):
        pass

# Renamed TransformerLayer to MistralLayer
class MistralLayer:
    def __init__(self, config, env, policy, i):
        self.attention = SelfAttention(config, env, policy, i)
        self.mlp = MLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        home1, home2 = weight_home.val
        # Get tensors from sub-layers and store them directly.
        attn_weights = self.attention.load_weight(home1, None, k)
        mlp_weights = self.mlp.load_weight(home2, None, k)

        if k == 0:
            # Store all tensors for this layer in a single tuple.
            weight_read_buf.store((attn_weights, mlp_weights))

    # forward logic is kept, as it just calls sub-layers
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        
        # 1. Unpack weights for both sub-layers.
        #    This follows the simplified structure from the corrected `load_weight`.
        if k == self.policy.num_gpu_batches - 1:
            # Pop from the buffer on the last GPU batch to consume it.
            attn_weights_tuple, mlp_weights_tuple = weight_read_buf.pop()
        else:
            # Peek at the buffer for other batches.
            attn_weights_tuple, mlp_weights_tuple = weight_read_buf.val

        # Self-Attention Block
        self.attention.forward(hidden, cache_read_buf, attn_weights_tuple, attention_mask,
                               cache_write_buf, i, k)

        # MLP Block 
        self.mlp.forward(hidden, None, mlp_weights_tuple, attention_mask, None, i, k)

    # Other methods like init_cache, load_cache, store_cache are delegated to SelfAttention
    def init_cache_one_gpu_batch(self, cache_home):
        self.attention.init_cache_one_gpu_batch(cache_home)
    def load_cache(self, cache_home, cache_read_buf, i):
        self.attention.load_cache(cache_home, cache_read_buf, i)
    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)


class MistralLM:
    
    config_class = MistralConfig

    def __init__(self,
                 config: MistralConfig,
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy,
                 model_id: str):
        #if isinstance(config, str):
        #    config = get_mistral_config(config)
        self.config = config

        self.model_id = model_id
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            layers.append(MistralLayer(self.config, self.env, self.policy, i))
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)

        self.weight_read_buf = array_1d(num_layers, ValueHolder)

        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()
        #self.diagnose_weights() for debugging model weights init

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def diagnose_weights(self):
        print("\n--- Running Weight Diagnostics ---")
        try:
            layer_0_weights = self.weight_home[1].val[0].val # MistralLayer -> SelfAttention -> weights
            q_proj_weight = layer_0_weights[1] # w_q

            weight_tensor = q_proj_weight.smart_copy(self.env.cpu)[0].data
            
            if torch.isnan(weight_tensor).any():
                print("!!! CRITICAL: NaN found in weights !!!")
            elif torch.isinf(weight_tensor).any():
                print("!!! CRITICAL: Inf found in weights !!!")
            elif weight_tensor.abs().mean().item() < 1e-6:
                print(f"!!! WARNING: Weights mean absolute value is very low ({weight_tensor.abs().mean().item()}). May indicate loading issue.")
            else:
                print("Weight diagnostics passed. Basic stats for layer 0 q_proj:")
                print(f"  - Mean: {weight_tensor.mean().item():.4f}")
                print(f"  - Std:  {weight_tensor.std().item():.4f}")
                print(f"  - Abs Mean: {weight_tensor.abs().mean().item():.4f}")
        except Exception as e:
            print(f"!!! ERROR during weight diagnostics: {e} !!!")
        print("--------------------------------\n")

    def init_weight(self, j, path):
        self.layers[j].init_weight(self.weight_home[j], path)

    def load_weight(self, i, j, k, overlap=True):
        if j == self.num_layers:
            j = 0; i += 1
            if i == self.execute_gen_len: return
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop(): y.delete()
                else: x.delete()

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        if i == 0: return
        if k == self.num_gpu_batches: k = 0; j += 1
        if j == self.num_layers:
            j = 0; i += 1
            if i == self.execute_gen_len: return
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        if k == -1: k = self.num_gpu_batches - 1; j -= 1
        if j == -1:
            j = self.num_layers - 1; i -= 1
            if i == -1: return
        if i == self.task.gen_len - 1:
            self.cache_write_buf[j][k].pop()
            return
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v: x.delete()

    def load_hidden(self, i, j, k):
        if k == self.num_gpu_batches: k = 0; j += 1
        if j == self.num_layers:
            j = 0; i += 1
            if i == self.execute_gen_len: return
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        if k == -1: k = self.num_gpu_batches - 1; j -= 1
        if j == -1:
            j = self.num_layers - 1; i -= 1
            if i == -1: return
        
        if j == self.num_layers - 1:
            # This is the final layer, process the output logits
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size

            # Pop the final TorchTensor (which contains logits on GPU)
            final_hidden_tensor = self.hidden[i][j][k].pop()

            # Get the raw torch.Tensor data and move to CPU for numpy conversion
            ids = final_hidden_tensor.data.detach().cpu().numpy()
            
            # Explicitly delete the GPU tensor after use
            final_hidden_tensor.delete()

            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:
            # This is an intermediate layer, move hidden state to its home device (CPU/Disk)
            x = self.hidden[i][j][k]
            if x.val:
                # Get the TorchTensor to be moved
                tensor_to_move = x.val

                # Use the copy() method to create a new tensor on the home device.
                # This creates a true, deep copy of the data on the target device.
                new_tensor_on_home = tensor_to_move.copy(self.act_home)

                # Replace the old tensor in the ValueHolder with the new one
                x.val = new_tensor_on_home

                # Delete the original GPU tensor to free VRAM
                tensor_to_move.delete()
                #x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k],
            self.cache_write_buf[j][k], i, k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        print('------ Start init all weights ------')
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        
        expanded_path = os.path.expanduser(f"~/flexgen_weights/{self.model_id}-np")
        
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(
                f"Weight directory not found at {expanded_path}. "
                "Please run 'python scripts/doiwnload_and_convert_weights.py' first."
            )

        for j in range(self.num_layers):
            self.init_weight(j, expanded_path)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        # For LLaMA/Mistral architecture, the attention mask is only needed
        # for the prefill stage (i=0) to handle padding. In the decoding stage (i > 0),
        # the causal attention is implicitly handled by the model architecture and
        # the position_ids passed to RoPE. We do not need to extend the mask.
        if i > 0:
            if self.attention_mask[k].val is not None:
                self.attention_mask[k].val.delete()
                self.attention_mask[k].clear()
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        
        padded_len = self.task.padded_len
        input_ids = self.output_ids[left:right, :padded_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute else self.env.gpu)
        val = attention_compute.allocate((self.policy.gpu_batch_size, padded_len), bool)

        mask_np = (input_ids != self.config.pad_token_id)
        val.load_from_np(mask_np)
        self.attention_mask[k].store(val)

    def generate(self, inputs: Union[np.array, List[List[int]]], max_new_tokens: int = 32,
                 do_sample: bool = False, temperature: float = 1.0, stop: Optional[int] = None,
                 verbose: int = 0):

        # 1. Clear all buffers at the beginning of each generation call.
        # This ensures that values from previous calls (like warmup) do not interfere.

        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
            for k in range(num_gpu_batches):
                # self.cache_home is for persistent cache, should not be cleared here.
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()

        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()

        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')
            prompt_len = int(np.sum(attention_mask[0])) # actual length
            padded_len = input_ids.shape[1]             # including padding
        else:
            #list or numpy array
            input_ids = np.asarray(inputs)
            attention_mask = (input_ids != self.config.pad_token_id)
            prompt_len = int(np.sum(attention_mask[0]))
            padded_len = input_ids.shape[1]

        task = Task(inputs=input_ids, prompt_len=prompt_len, padded_len=padded_len,
                    gen_len=max_new_tokens, cut_gen_len=None,
                    do_sample=do_sample, temperature=temperature, stop=stop)

        self.execute_gen_len = task.gen_len
        self.output_ids = np.full((len(task.inputs), padded_len + task.gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :padded_len] = task.inputs

        assert self.policy.gpu_batch_size * self.policy.num_gpu_batches == len(task.inputs)
        
        for j in range(self.num_layers):
            for k in range(self.policy.num_gpu_batches):
                self.cache_home[j][k].clear()
        self.hidden = array_3d(task.gen_len, self.num_layers, self.policy.num_gpu_batches, ValueHolder)
        self.set_task(task)
        for j in range(self.num_layers):
            for k in range(self.policy.num_gpu_batches):
                self.init_cache(j, k)

        # Simplified generation loop for clarity
        for i in tqdm(range(self.execute_gen_len), desc="Generating tokens"):
            timers("generate").start()
            
            # Synchronize before starting a new generation step.
            self.sync()

            for k in range(self.policy.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                # Load weights (already synchronous in nature)
                self.load_weight(i, j, 0, overlap=False)

                # Load cache and hidden states
                self.load_cache(i, j, 0, overlap=False)
                self.load_hidden(i, j, 0)
                
                # Synchronize before computation.
                # Ensures all data (weights, cache, hidden) is fully loaded and ready on the GPU.
                self.sync()

                self.compute_layer(i, j, 0)

                # Synchronize after computation.
                # Ensures the forward pass is complete before moving data around.
                self.sync()
                
                self.store_hidden(i, j, 0)
                self.store_cache(i, j, 0, overlap=False)

                # Synchronize after storing data.
                self.sync()
            timers("generate").stop()

        for j in range(self.num_layers):
            for k in range(self.policy.num_gpu_batches):
                self.delete_cache(j, k)
        return self.output_ids

def get_inputs(prompt_len, num_prompts, tokenizer, path):
    prompts = []
    with open(path, 'r') as file:
        prompts.append(file.read())
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    input_ids[0] = input_ids[0][:prompt_len]
    return (input_ids[0],) * num_prompts

def run_flexgen(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len = args.prompt_len, args.gen_len
    warmup_inputs = get_inputs(256, num_prompts, tokenizer, args.warmup_input_path)
    inputs = get_inputs(prompt_len, num_prompts, tokenizer, args.test_input_path)
    
    mistral_config = get_mistral_config(args.model)
    gpu = TorchDevice("cuda:0", config=mistral_config)
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1], args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5], False, args.pin_weight, False,
                    args.compress_weight, CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
                    args.compress_cache, CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False))
    
    model = MistralLM(mistral_config, env, args.path, policy, model_id=args.model)
    print('------ Model Load Complete ------')

    try:
        output_ids = model.generate(warmup_inputs, max_new_tokens=1, verbose=args.verbose)

        timers("generate").reset()
        output_ids = model.generate(inputs, max_new_tokens=args.gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    
    decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    print("\n--- Inference Performance ---")
    print(f"Model: {args.model}")
    print(f"Batch Size: {num_prompts}, Prompt Length: {prompt_len}, Generation Length: {args.gen_len}")
    print(f"GPU Peak Memory: {gpu_peak_mem / GB:.3f} GB")
    print("---------------------------------")
    print(f"Prefill Latency: {prefill_latency:.3f} s")
    print(f"Prefill Throughput: {prefill_throughput:.1f} tokens/s")
    print("---------------------------------")
    print(f"Decode Latency: {decode_latency:.3f} s")
    print(f"Decode Throughput: {decode_throughput:.1f} tokens/s")
    print("---------------------------------")
    print(f"Total Latency: {total_latency:.3f} s")
    print(f"Total Throughput: {total_throughput:.1f} tokens/s")
    print("=================================")
    print("\n")

def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)

    parser.add_argument("--warmup-input-path", type=str)
    parser.add_argument("--test-input-path", type=str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexgen(args)
