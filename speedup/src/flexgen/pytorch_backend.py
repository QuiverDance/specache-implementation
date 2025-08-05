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
# Modified by QuiverDance int 2025: convert OPT into Mistral
# © 2025 QuiverDance

from enum import Enum, auto
from itertools import count
import os
import queue
import shutil
import threading
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from flexgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None

def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice

class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and compressed tensors on
        GPUs, CPUs, disks and mixed devices.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
      type: Tuple[Tuple[TorchTensor], Tuple[int]]
    For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
      type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_np(self, np_array):
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                tmp = torch.from_numpy(np_array)
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                torch_tensor = torch.from_numpy(np_array)
                self.data.copy_(torch_tensor)

    def load_from_np_file(self, filename):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            loadded_np = np.load(filename)
            self.load_from_np(loadded_np)

    def copy(self, dst, src_indices=None):
        if src_indices:
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
        else:
            shape = self.shape

        if dst.device_type == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, weight_tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (weight_tensor * hidden_states).to(input_dtype)

#RoPE(Rotary Position Embedding)
class MistralRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None, config=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = TorchCompressedDevice(self)

        self.links = {}

        self.attention_compute_workspace = None
        self.workspace_pt = 0

        # init for RoPE cache:
        if config:
            head_dim = config.hidden_size // config.num_attention_heads
            # The key is `device=self.dev`, which will be 'cpu' for the cpu_device instance
            # and 'cuda:0' for the gpu_device instance.
            self.rotary_emb = MistralRotaryEmbedding(
                dim=head_dim, 
                max_position_embeddings=config.max_position_embeddings, 
                base=10000, 
                device=self.dev
            )
            self.rmsnorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    # Input Embedding (Simplifed)
    def mistral_input_embed(self, inputs, w_token):
        token_ids = inputs.data
        
        data = F.embedding(token_ids, w_token.data)
        
        return TorchTensor.create_from_torch(data, self)

    # Output Embedding
    def mistral_output_embed(self, inputs, w_ln, w_lm_head,
                             do_sample, temperature, rms_norm_eps):
        if w_lm_head.device.device_type == DeviceType.COMPRESSED:
            w_lm_head = w_lm_head.device.decompress(w_lm_head)

        # RMSNorm
        hidden = self.rmsnorm(inputs.data, w_ln.data)

        logits = F.linear(hidden, w_lm_head.data)
        last_token_logits = logits[:, -1, :]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=-1, keepdim=True)
        
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch(self, config, task, policy):
        """
        Allocates a KV cache for one GPU batch on this device.
        """
        # Calculating cache shape considering GQA
        batch_size = policy.gpu_batch_size
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        
        # Allocate cache only for the required length, not the maximum possible length.
        if task:
            required_len = task.prompt_len + task.gen_len
        else:
            required_len = 512
            
        # Define cache shape: [batch, num_kv_heads, required_len, head_dim]
        shape = (batch_size, num_kv_heads, required_len, head_dim)
        
        # pin_memory can have a large memory overhead, so set it to False.
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        
        return k_cache, v_cache

    def mha(self, inputs, attention_mask, w_ln, w_q, w_k, w_v, w_out,
            config, donate, compress_cache, comp_config):
        """
        Multi-Head Attention for prefill stage.
        """
        # 0. Store a reference to the input tensor for the residual connection.
        residual_connection = inputs.data

        b, s, h = inputs.shape
        head_dim = h // config.num_attention_heads
        
        # 1. RMS Normalization
        hidden = self.rmsnorm(inputs.data, w_ln.data)
        
        # 2. Q, K, V Projection
        q = F.linear(hidden, w_q.data)
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        
        # 3. Reshape to heads
        q = q.view(b, s, config.num_attention_heads, head_dim).transpose(1, 2)
        k = k.view(b, s, config.num_key_value_heads, head_dim).transpose(1, 2)
        v = v.view(b, s, config.num_key_value_heads, head_dim).transpose(1, 2)
        
        # 4. Apply RoPE
        target_device = inputs.device.dev
        position_ids = torch.arange(0, s, dtype=torch.long, device=target_device).unsqueeze(0)
        cos, sin = self.rotary_emb(v, seq_len=s)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # 5. Create cache from a copy of ORIGINAL k and v, BEFORE GQA repeat.
        # The shape of the cache should be [batch, num_kv_heads, seq_len, head_dim].
        k_cache_to_return = TorchTensor.create_from_torch(k.clone(), self)
        v_cache_to_return = TorchTensor.create_from_torch(v.clone(), self)

        # 6. Apply GQA by repeating K, V heads for attention calculation only
        n_rep = config.num_attention_heads // config.num_key_value_heads
        k_for_attn = self.repeat_kv(k, n_rep)
        v_for_attn = self.repeat_kv(v, n_rep)
        
        # 7. Caluate Attention
        causal_mask = torch.triu(torch.ones(s, s, device=target_device, dtype=torch.bool), diagonal=1)
        inverted_padding_mask = (attention_mask.data == 0).view(b, 1, 1, s)
        final_mask = inverted_padding_mask | causal_mask

        value = torch.nn.functional.scaled_dot_product_attention(
            q, k_for_attn, v_for_attn, attn_mask=final_mask
        )

        # 8. Output Projection and Residual Connection
        value = value.transpose(1, 2).contiguous().view(b, s, h)
        value = F.linear(value, w_out.data)
        value.add_(residual_connection)
        
        return TorchTensor.create_from_torch(value, self), k_cache_to_return, v_cache_to_return

    # GQA helper function
    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def mha_gen(self, inputs, w_ln, w_q, w_k, w_v, w_out, config,
                k_cache, v_cache, donate, compress_cache, comp_config):
        """
        Multi-Head Attention for decoding stage.
        """
        # 0. Store a reference to the input tensor for the residual connection.
        residual_connection = inputs.data

        b, s, h = inputs.shape # s is 1
        head_dim = h // config.num_attention_heads
        
        # 1. RMS Normalization
        hidden = self.rmsnorm(inputs.data, w_ln.data)

        # 2. Q, K, V Projection
        q = F.linear(hidden, w_q.data)
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        
        # 3. Reshape to heads
        q = q.view(b, s, config.num_attention_heads, head_dim).transpose(1, 2)
        k = k.view(b, s, config.num_key_value_heads, head_dim).transpose(1, 2)
        v = v.view(b, s, config.num_key_value_heads, head_dim).transpose(1, 2)
        
        # 4. Apply RoPE
        # k_cache shape is [b, num_kv_h, seq_len, h_dim].
        past_key_values_length = k_cache.shape[2] 
        position_ids = torch.tensor([[past_key_values_length]], dtype=torch.long, device=self.dev)
        
        cos, sin = self.rotary_emb(v, seq_len=past_key_values_length + 1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # 5. Concatenate current K, V to past cache
        k_cache_updated = torch.cat([k_cache.data, k], dim=2)
        v_cache_updated = torch.cat([v_cache.data, v], dim=2)
        
        # 6. Apply GQA
        n_rep = config.num_attention_heads // config.num_key_value_heads
        k_for_attn = self.repeat_kv(k_cache_updated, n_rep)
        v_for_attn = self.repeat_kv(v_cache_updated, n_rep)
        
        # 7. Attention Score Calculation and Value
        attn_weights = torch.matmul(q, k_for_attn.transpose(2, 3)) / np.sqrt(head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        value = torch.matmul(attn_weights, v_for_attn)

        # 8. Output Projection and Residual Connection
        value = value.transpose(1, 2).contiguous().view(b, s, h)
        value = F.linear(value, w_out.data)
        value.add_(residual_connection)
        
        # 9. Return new value and the delta of the cache
        k_cache_new = TorchTensor.create_from_torch(k, self)
        v_cache_new = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k_cache_new, v_cache_new

    # SwiGLU MLP
    def mistral_mlp(self, inputs, w_ln, w_gate, w_up, w_down, rms_norm_eps, donate):
        # 1. Store a reference to the input tensor for the residual connection.
        #    We directly use the tensor data, which will not be affected by
        #    the deletion of the wrapper object.
        residual_connection = inputs.data

        # 2. Perform RMS Normalization.
        h = self.rmsnorm(inputs.data, w_ln.data)
        
        # 3. We NO LONGER need to worry about `inputs` being deleted, but as a
        #    precaution, let's handle the `donate` flags for weights only.
        if donate[1]: w_ln.delete()

        # 4. MLP calculation
        gate = F.linear(h, w_gate.data)
        if donate[2]: w_gate.delete()
        
        up = F.linear(h, w_up.data)
        if donate[3]: w_up.delete()
        
        fused_mlp = F.silu(gate) * up
        
        down = F.linear(fused_mlp, w_down.data)
        if donate[4]: w_down.delete()
        
        # 5. Perform the residual connection using the saved tensor.
        #    This is now safe from any potential side effects.
        down.add_(residual_connection)
        
        # We can now safely delete the original input wrapper if requested.
        if donate[0]: inputs.delete()

        # 6. Return the result wrapped in a new TorchTensor.
        return TorchTensor.create_from_torch(down, self)
    
    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"
    
    def allocate(self, shape, dtype, pin_memory=None, name=None):
        """
        Allocates an empty tensor on this device.
        """
        if self.device_type == DeviceType.CPU:
            # pin_memory is only relevant for CPU tensors to speed up CPU->GPU copies.
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False

        torch_dtype = np_dtype_to_torch_dtype[dtype]
        data = torch.empty(shape, dtype=torch_dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=0):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.compressed_device = TorchCompressedDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}

        # Copy threads
        self.copy_queue = None
        self.copy_threads = []
        if num_copy_threads > 0:
            self.copy_queue = queue.Queue()
            self.copy_threads = [
                threading.Thread(
                    target=copy_worker_func, args=(self.copy_queue, cuda_id)
                ) for _ in range(num_copy_threads)
            ]
            for t in self.copy_threads:
                t.start()
        
        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.num_attention_heads, config.hidden_size, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache

    def submit_copy(self, *args):
        if self.copy_queue is None:
            self._copy_sync(*args)
        else:
            self.copy_queue.put_nowait(args)

    def synchronize(self):
        if self.copy_queue is not None:
            self.copy_queue.join()
        #self.copy_queue.join()

    def close_copy_threads(self):
        if self.copy_queue is not None:
            for _ in range(len(self.copy_threads)):
                self.copy_queue.put_nowait(None)
            for t in self.copy_threads:
                t.join()
            self.copy_queue.join()
            self.copy_queue = None
    
    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()

    def _copy_sync(self, dst, dst_indices, src, src_indices):
        """A function that copies data synchronously."""
        src_data = map_to_torch_tensor(src, src_indices)
        dst_data = map_to_torch_tensor(dst, dst_indices)
    
        # Handle the case of copying CPU tensors to GPU
        if (isinstance(dst_data, torch.Tensor) and dst_data.is_cuda and
            isinstance(src_data, np.ndarray)):
            # numpy -> torch (cpu) -> torch (gpu)
            dst_data.copy_(torch.from_numpy(src_data))
        else:
            # Other cases (e.g., disk -> cpu)
            dst_data.copy_(src_data)

# Segment dimension for tensors stored on TorchMixedDevice
SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.num_attention_heads, config.hidden_size, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """An I/O link between two devices."""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        if force_io_time is not None:
            return force_io_time

        return size / bandwidth


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if dst.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert src.device.device_type != DeviceType.MIXED
        seg_points = dst.data[1]

        for i in range(len(dst.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    elif src.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory
        # as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        #src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)


def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()
