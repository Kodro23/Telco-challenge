import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import torch

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder


def parse_arge():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        help="add any help statement if you want(optional parameter)",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="lm_dataset", help="add any help statement if you want(optional parameter)"
    )
    parser.add_argument(
        "--hf_token", type=str, default="<ADD_YOUR_HF_TOKEN>", help="add any help statement if you want(optional parameter)"
    )
  
    parser.add_argument(
        "--epochs", type=int, default=3, help="add any help statement if you want(optional parameter)"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="add any help statement if you want(optional parameter)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="add any help statement if you want(optional parameter)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="add any help statement if you want(optional parameter)"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="add any help statement if you want(optional parameter)",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="add any help statement if you want(optional parameter)",
    )
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="add any help statement if you want(optional parameter)",
    )
    args, _ = parser.parse_known_args()

    if args.hf_token:
       print("logging into hf hub with token")
       login(token=args.hf_token)

    return args



def print_trainable_parameters(model, use_4bit=False):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )



