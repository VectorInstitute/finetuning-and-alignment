from __future__ import annotations

import math
from typing import Any

import torch
from datasets import load_dataset
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    PreTrainedTokenizer,
)


class DataCollatorWithPadding:
    """Data collator for preprocessing a batch.

    Similar to the one offered by HF, but here we can keep track of any extra
    keys.

    Attributes
    ----------
        pad_token_id: A token id that is used for padding.
        ignore_index: A value used for ignoring a given token in labels.
        max_seq_len: An integer denoting the maximum sequence length.
        padding_side: A side of the sequence that gets padded.

    """

    def __init__(
        self,
        pad_token_id: int,
        ignore_index: int,
        max_seq_len: int,
        padding_side: str,
    ) -> None:
        """Initialize the data collator instance.

        Args:
        ----
            pad_token_id: The token id used for padding.
            ignore_index: The index used to ignore labels while calculating
                loss.
            max_seq_len: The maximum sequence length to expect.
            padding_side: The side of the sequence which is padded.

        """
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_seq_len = max_seq_len
        self.padding_side = padding_side

    def __call__(
        self,
        instances: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        """Create an input batch.

        Convert incoming tokenized instances to the relevant batch items. The
        batch contains `id` which is the unique id for a training data point,
        `input_ids` which are tokens in the sequence with padding applied,
        `labels` which are the predicted tokens with ignore index applied,
        and `attention_mask` which dictates which tokens will be masked out
        during the self-attention mechanism.

        Args:
        ----
            instances: A list containing dictionary datapoints in a HF format.

        Returns:
        -------
            A dictionary containing a batch that we can input to our model.

        """
        batch = {}
        keys = ["input_ids", "labels"]
        input_ids, labels = tuple(
            [
                torch.tensor(
                    instance[key][0 : self.max_seq_len],
                )
                for instance in instances
            ]
            for key in keys
        )

        if self.padding_side == "left":
            input_ids = self._reverse_tensor(input_ids)
            labels = self._reverse_tensor(labels)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.ignore_index,
        )

        if self.padding_side == "left":
            input_ids = input_ids.flip(dims=(1,))
            labels = labels.flip(dims=(1,))

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["attention_mask"] = batch["input_ids"].ne(self.pad_token_id)
        return batch


    def _reverse_tensor(
        self,
        x: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Tensors in `x` have shape (S,)."""
        return [t.flip(0) for t in x]


def create_prompt_formats_it(
    samples: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, list[list[int]]]:
    """Tokenizes the incoming data samples to follow the IFT structure."""
    instruction = (
        "<s> [INST] <<SYS>> You are a text debiasing bot, you take as input a"
        " text and you output its debiased version by rephrasing it to be"
        " free from any age, gender, political, social or socio-economic"
        " biases, without any extra outputs. Debias this text by rephrasing"
        " it to be free of bias: <</SYS>> "
    )
    all_input_ids = []
    all_labels = []
    all_attention_mask = []
    for biased_text, debiased_text in zip(
        samples["biased_text"],
        samples["debiased_text"],
    ):
        input_context = f"{biased_text} [/INST] " # column with biased input
        response = f"{debiased_text}" # column with debiased output
        end = " </s>"
        first_half = "".join([
            part for part in [instruction, input_context] if part
        ])
        second_half = "".join([
            part for part in [response, end] if part
        ])
        labels = []
        all_input_ids.append(
            tokenizer.encode(
                first_half,
                add_special_tokens=False,
            ) + tokenizer.encode(
                second_half,
                add_special_tokens=False,
            ),
        )
        all_attention_mask.append([1] * len(all_input_ids[-1]))
        labels.extend(
            [-100] * len(
                tokenizer.encode(first_half, add_special_tokens=False),
            ),
        )
        labels.extend(tokenizer.encode(second_half, add_special_tokens=False))
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_mask,
    }


def get_dataloaders(
    ds_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, int]:
    """Get train and eval dataloaders.

    Args:
    ----
        ds_path: The path to the saved dataset. Expected to be a CSV.
        tokenizer: The tokenizer.
        batch_size: The train/eval batch size.

    Returns:
    -------
        The train, eval dataloaders, and unsharded length of train set.

    """
    # load the dataset and select the correct columns
    full_ds = load_dataset("csv", data_files=ds_path)
    full_ds = full_ds["train"].train_test_split(test_size=0.1)
    full_ds = full_ds.select_columns(["biased_text", "debiased_text"])

    # tokenize the dataset
    tokenized_train = full_ds["train"].map(
        lambda examples: create_prompt_formats_it(examples, tokenizer),
        batched=True,
        batch_size=250,
        num_proc=4,
        remove_columns=full_ds["train"].column_names,
    )
    tokenized_test = full_ds["test"].map(
        lambda examples: create_prompt_formats_it(examples, tokenizer),
        batched=True,
        batch_size=250,
        num_proc=4,
        remove_columns=full_ds["test"].column_names,
    )
    original_length = math.ceil(len(tokenized_train) / batch_size)

    # instantiate the data collator
    dc = DataCollatorWithPadding(
        tokenizer.pad_token_id,
        -100,
        tokenizer.model_max_length,
        tokenizer.padding_side,
    )

    # distribute the correct chunks of data to the respective workers (GPUs)
    train_sampler = DistributedSampler(
        tokenized_train,
        dist.get_world_size(),
        dist.get_rank(),
        shuffle=True,
    )
    test_sampler = DistributedSampler(
        tokenized_test,
        dist.get_world_size(),
        dist.get_rank(),
        shuffle=False,
    )

    # instantiate the data loaders
    train_dataloader = DataLoader(
        tokenized_train,
        collate_fn=dc,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
    )
    eval_dataloader = DataLoader(
        tokenized_test,
        collate_fn=dc,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
    )
    return train_dataloader, eval_dataloader, original_length
