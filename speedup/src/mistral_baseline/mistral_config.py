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

"""
The Mistral model configurations.
"""

import dataclasses
import numpy as np

@dataclasses.dataclass(frozen=True)
class MistralConfig:
    name: str = "mistral-7b"
    num_hidden_layers: int = 32
    max_seq_len: int = 32768  # Mistral-7B's context window
    hidden_size: int = 4096
    n_head: int = 32
    # For Grouped-Query Attention
    num_key_value_heads: int = 8
    input_dim: int = 4096
    ffn_embed_dim: int = 14336
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-5
    pad_token_id: int = 0  # Mistral uses 0 for padding
    dtype: type = np.float16

    def model_bytes(self):
        # A rough estimation
        return self.num_hidden_layers * (
            # Self-Attention
            self.hidden_size * self.hidden_size * 4 +  # Q, K, V, O projections
            # MLP (Gated)
            self.hidden_size * self.ffn_embed_dim * 3 + # gate, up, down projections
            # LayerNorm
            self.hidden_size * 2 * 2  # Input and Post-Attention RMSNorm
        ) * 2 + (self.vocab_size * self.hidden_size * 2) # Embeddings

    def cache_bytes(self, batch_size, seq_len):
        # For GQA, K and V caches are smaller
        key_cache_size = self.num_key_value_heads * (self.hidden_size // self.n_head)
        value_cache_size = key_cache_size
        return 2 * batch_size * seq_len * self.num_hidden_layers * (key_cache_size + value_cache_size) * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def get_mistral_config(name, **kwargs):
    if "7b" in name.lower():
        config = MistralConfig()
    else:
        raise ValueError(f"Invalid model name for Mistral: {name}")

    return dataclasses.replace(config, **kwargs)
