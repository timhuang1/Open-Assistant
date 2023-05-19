#!/usr/bin/env python3
import argparse
import logging
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import torch

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
print(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
print(parent_dir)


# from model_training.custom_datasets.formatting import DatasetEntry
from model_training.custom_datasets.dialogue_collator import DialogueDataCollator
from model_training.efficiency_utils import fuse_gelu
from model_training.utils.utils import (
    PerDatasetSampler,
    _strtobool,
    get_dataset,
    get_loss,
    get_metrics,
    get_model,
    get_tokenizer,
    init_rng,
    read_yamls,
)
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker
from transformers.training_args import OptimizerNames
from transformers.utils import is_datasets_available


def argument_parsing(notebook=False, notebook_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="""
        Multiple configs can be passed to set different options.
        For example, run as:

           ./trainer_sft.py --configs galactica-125m webgpt_dataset_only per_digit_tokens

        to run the galactica-125m model, using the webgpt dataset only (as opposed to all
        the datasets listed in defaults in config.yaml) and treat each digit as a separate token.
    """,
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--no-deepspeed", dest="deepspeed", action="store_false")
    parser.add_argument("--wandb-entity", type=str, default="open-assistant")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from last saved checkpoint")
    parser.add_argument("--rng_seed", type=int, help="rng seed")
    parser.add_argument("--show_dataset_stats", action="store_true", help="Show dataset stats", default=False)
    parser.set_defaults(deepspeed=False)

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("./configs")
    conf.update(configs["defaults"])
    try:
        for name in args.configs:
            if "," in name:
                for n in name.split(","):
                    conf.update(configs[n])
            else:
                conf.update(configs[name])
    except KeyError as e:
        print(f'Error: Could not find the config "{e.args[0]}" in config.yaml')
        exit(1)

    conf["wandb_entity"] = args.wandb_entity
    conf["local_rank"] = args.local_rank
    conf["deepspeed"] = args.deepspeed
    conf["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.rng_seed is not None:
        conf["rng_seed"] = args.rng_seed
    conf["show_dataset_stats"] = args.show_dataset_stats

    # get the world size in deepspeed
    if conf["deepspeed"]:
        conf["world_size"] = int(os.getenv("WORLD_SIZE", default="1"))
    else:
        conf["world_size"] = 1

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)
        # Allow --no-{key}  to remove it completely
        parser.add_argument(f"--no-{key}", dest=key, action="store_const", const=None)

    return parser.parse_args(remaining)


def main():
    training_conf = argument_parsing()
    if not training_conf.deepspeed or training_conf.local_rank == 0:
        print(f"trainig_conf = {training_conf}")

    if training_conf.val_max_length is None:
        training_conf.val_max_length = training_conf.max_length

    train, evals = get_dataset(training_conf)

    show_dataset_stats = (training_conf.verbose or training_conf.show_dataset_stats) and (
        not training_conf.deepspeed or training_conf.local_rank == 0
    )
    if show_dataset_stats:
        print("Training dataset sizes (before sampling):")
        total = len(train)
        for d in train.datasets:
            if isinstance(d, Subset):
                name = f"Subset of {type(d.dataset).__name__}"
                if hasattr(d.dataset, "name"):
                    name += f" ({d.dataset.name})"
            else:
                name = type(d).__name__
                if hasattr(d, "name"):
                    name += f" ({d.name})"
            print(f"{name}: {len(d)} ({len(d) / total:.2%})")

            # ensure that all entries can be formatted
            # for x in d:
            #     if isinstance(x, DatasetEntry):
            #         x.get_formatted("sft", "<eos>")

        print(f"\nTotal train: {total}")
        print("-" * 80)
        print("Evaluation set sizes:")
        total_eval = sum(len(x) for x in evals.values())
        for k, d in evals.items():
            print(f"{k}: {len(d)} ({len(d) / total_eval:.2%})")
        print(f"\nTotal eval: {total_eval}")
        print("-" * 80)


if __name__ == "__main__":
    main()
