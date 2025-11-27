from __future__ import annotations

import torch
from torch import distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def setup() -> None:
    """Initialize the process group and create directories."""
    dist.init_process_group("nccl")


def cleanup() -> None:
    """Clean up the process group after training."""
    dist.destroy_process_group()


def load_model_and_tokenizer(
    path: str,
    use_mp: bool,
    use_fa: bool,
    max_seq_len: int | None = None,
    use_safetensors: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the model and tokenizer.

    Args:
    ----
        path: The path where the model and tokenizer are stored.
        use_mp: Whether to use mixed-precision.
        use_fa: Whether to use Flash Attention 2.
        max_seq_len: The maximum sequence length.
        use_safetensors: Whether to use HF safe tensors. Note that this format
            loads significantly faster.

    Returns:
    -------
        The model and tokenizer.


    """
    # load model
    model_args = {"use_cache": False, "use_safetensors": use_safetensors}
    model_args["torch_dtype"] = torch.bfloat16
    if use_fa:
        if not use_mp:
            msg = "Use FA with bf16 (mixed precision)"
            raise ValueError(msg)
        model_args["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        path,
        **model_args,
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if max_seq_len:
        tokenizer.model_max_length = max_seq_len

    # extend embeddings to a multiple so we use Tensor cores
    multiple = 64 if "A100" in torch.cuda.get_device_name() else 8
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=multiple,
    )
    return model, tokenizer
