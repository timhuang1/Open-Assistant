import argparse
import logging
import os
import copy
import datasets
import torch
import sys
import random
import numpy as np
from functools import partial
from itertools import chain
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datasets import Dataset, concatenate_datasets
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase, TruncationStrategy
from multiprocessing import Pool
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model_training.custom_datasets.formatting import (
    QA_SPECIAL_TOKENS,
    DatasetEntryLm,
    DatasetEntrySft,
    format_pairs,
    format_system_prefix,
)
from model_training.custom_datasets.dialogue_collator import DialogueDataCollator
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
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm


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

    conf["configs"] = [name for name in args.configs]
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


if __name__ == "__main__":
    training_conf = argument_parsing()
    if not training_conf.deepspeed or training_conf.local_rank == 0:
        print(f"trainig_conf = {training_conf}")

    if training_conf.val_max_length is None:
        training_conf.val_max_length = training_conf.max_length

    if "bloom_minus" in training_conf.model_name:
        import sys
        sys.path.insert(1, "/apdcephfs/share_916081/timxthuang/cond_gen_git/examples/pytorch")
        from utils import BloomMinusTokenizerFast
        tokenizer = BloomMinusTokenizerFast.from_pretrained(training_conf.model_name)
        additional_special_tokens = list(QA_SPECIAL_TOKENS.values())
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    else:
        tokenizer = get_tokenizer(training_conf)

    collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=training_conf.max_length,
        random_offset_probability=training_conf.random_offset_probability,
        label_masking=training_conf.label_masking,
        samples_mixing=training_conf.samples_mixing,
        mix_length_threshold=training_conf.mix_length_threshold,
        pad_to_multiple_of=16,
        use_system_prefix=training_conf.use_system_prefix,
        system_prefix=training_conf.system_prefix,
        use_system_tag=training_conf.use_system_tag,
        system_property_dropout=training_conf.system_property_dropout,
        system_add_length=training_conf.system_add_length,
    )

    if "get_local_dataset" in training_conf.configs:
        saved_files = [os.path.join(training_conf.local_dataset_dir, pt_filename) for pt_filename in training_conf.local_dataset_files]
        assert all([os.path.isfile(filename) for filename in saved_files]), f"Invalid pt files: {training_conf.local_dataset_files}"
        all_sub_ds = [torch.load(filename) for filename in saved_files]
        train = ConcatDataset(all_sub_ds)
    else:
        train, evals = get_dataset(training_conf)
    
    if "concat_save_to_local" in training_conf.configs:
        os.makedirs(training_conf.concat_save_dir, exist_ok=True)
        torch.save(train, os.path.join(training_conf.concat_save_dir, f"{training_conf.concat_save_subname}.pt"))
        # torch.save(evals, os.path.join(training_conf.concat_save_dir, f"{training_conf.concat_save_subname}_evals.pt"))

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
        print(f"\nTotal train: {total}")
        print("-" * 80)

        if "evals" in locals():
            print("Evaluation set sizes:")
            total_eval = sum(len(x) for x in evals.values())
            for k, d in evals.items():
                print(f"{k}: {len(d)} ({len(d) / total_eval:.2%})")
            
            print(f"\nTotal eval: {total_eval}")
            print("-" * 80)

    def raw_to_arraw(
        messages: Union[DatasetEntrySft, DatasetEntryLm]
    ) -> Dict[str, List[str]]:
        pretrain_dataset = False
        if isinstance(messages, DatasetEntrySft):
            messages = messages.get_formatted(
                eos_token=collate_fn.tokenizer.eos_token,
                use_system_tag=collate_fn.use_system_tag,
                system_property_dropout=collate_fn.system_property_dropout,
                system_add_length=collate_fn.system_add_length,
            )
        elif isinstance(messages, DatasetEntryLm):
            messages = [messages.text]
            pretrain_dataset = True
        else:
            messages = list(messages)
            messages = format_pairs(messages, collate_fn.tokenizer.eos_token)
        res = {"messages": messages, "pretrain_dataset": pretrain_dataset}
        return res

    def messages_tokenize_function(
        examples: Dict[str, List[List[Union[str, bool]]]]
    ) -> Dict[str, List[int]]:
        # details refer to DialogueDataCollator
        all_messages = examples["messages"]
        is_pretrain_labels = examples["pretrain_dataset"]
        res = defaultdict(list)
        for messages, is_pretrain in zip(all_messages, is_pretrain_labels):
            if random.random() < collate_fn.random_offset_probability\
                    and not is_pretrain:
                truncation = TruncationStrategy.DO_NOT_TRUNCATE
                max_length = None
            else:
                truncation = TruncationStrategy.LONGEST_FIRST
                max_length = collate_fn.max_length
            flatten_message = collate_fn.tokenizer(
                "".join(messages),
                max_length=max_length,
                truncation=truncation,
                padding=False,
            )
            if is_pretrain:
                label_mask = np.ones(len(flatten_message.input_ids), dtype=bool)
                # return flatten_message, label_mask, 0
                for k, v in flatten_message.items():
                    if k != "offset_mapping":
                        res[k].append(v)
                res["labels"].append(flatten_message["input_ids"])

                continue

            message_indices: Optional[list[int]] = None
            if collate_fn.label_masking:
                prompter_token_id = collate_fn.tokenizer.convert_tokens_to_ids(QA_SPECIAL_TOKENS["Question"])
                assistant_token_id = collate_fn.tokenizer.convert_tokens_to_ids(QA_SPECIAL_TOKENS["Answer"])
                assert prompter_token_id >= 0 and assistant_token_id >= 0
                message_indices = []
                i = -1
                for x in flatten_message.input_ids:
                    if x in (prompter_token_id, assistant_token_id):
                        i += 1
                    message_indices.append(i)
            input_length = len(flatten_message.input_ids)

            if collate_fn.max_length and input_length > collate_fn.max_length:
                offset = random.randint(0, input_length - collate_fn.max_length)
                for k in flatten_message.keys():
                    v = flatten_message[k]
                    if isinstance(v, list) and len(v) == input_length:
                        flatten_message[k] = v[offset : offset + collate_fn.max_length]
                if message_indices:
                    message_indices = message_indices[offset : offset + collate_fn.max_length]

            if collate_fn.label_masking:
                label_mask = np.array(list(map(lambda x: x % 2 == 1, message_indices)))
            else:
                label_mask = np.ones(len(flatten_message.input_ids), dtype=bool)

            label_mask[-1] = False  # make sure last token is inactive, has an effect only when truncating

            for k, v in flatten_message.items():
                if k != "offset_mapping":
                    res[k].append(v)
            # res["label_mask"].append(label_mask)
            # res["labels"] = [[-100] * len(src_ids) + tgt_ids for src_ids, tgt_ids in zip(model_inputs["input_ids"], labels["input_ids"])]
            labels = np.array(flatten_message.input_ids)
            labels[~label_mask] = -100
            res["labels"].append(labels)
        return res

    def pairwise_group_texts(
        examples: Dict[str, List[List[Union[str, bool]]]]
    ) -> Dict[str, List[int]]:
        # get all length
        sample_length = [len(ids) for ids in examples["input_ids"]]
        
        def get_chunks(numbers, N):
            current_sum, start_idx, last_idx = 0, 0, None
            segs = list()
            for idx, number in enumerate(numbers):
                if current_sum + number <= N:
                    current_sum += number
                else:
                    last_idx = idx
                    segs.append((start_idx, last_idx))
                    start_idx = idx
                    current_sum = number
            if last_idx is not None and last_idx < len(numbers):
                segs.append((last_idx, len(numbers)))
            return segs

        chunk_segs = get_chunks(sample_length, collate_fn.max_length)
        # print(f"chunk_segs: {chunk_segs}")
        result = {k: [] for k in examples.keys()}
        for start_idx, last_idx in chunk_segs:
            for k in examples.keys():
                result[k].append(list(chain(*examples[k][start_idx: last_idx])))
        return result

    def multiprocess_helper(index):
        # This function will be run in a separate process
        data = train[index]
        return raw_to_arraw(data)

    if "ds_save_to_local" in training_conf.configs:
        # dict_batch = raw_to_arraw(train)
        with Pool(training_conf.preprocessing_num_workers) as pool:
            # Use the pool's map function to apply process_data to each index in the dataset
            processed_dataset = list(pool.map(multiprocess_helper, range(len(train))))
        dataset = Dataset.from_list(processed_dataset)
        dataset = dataset.shuffle()
        tokenized_datasets = dataset.map(
            messages_tokenize_function,
            batched=True,
            # num_proc=training_conf.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        if collate_fn.samples_mixing:
            short_dataset = tokenized_datasets.filter(lambda x: len(x["input_ids"]) <= collate_fn.mix_length_threshold)
            long_dataset = tokenized_datasets.filter(lambda x: len(x["input_ids"]) > collate_fn.mix_length_threshold)
            split_res = short_dataset.train_test_split(test_size=collate_fn.mix_probability, shuffle=True)
            short_only_ds, short_pack_ds = split_res["train"], split_res["test"]

            pack_short_dataset = short_pack_ds.map(
                pairwise_group_texts,
                batched=True,
                # num_proc=1,
                load_from_cache_file=False,
                desc="Grouping texts in chunks",
            )
            tokenized_datasets = concatenate_datasets([long_dataset, short_only_ds, pack_short_dataset])
        tokenized_datasets.save_to_disk(os.path.join(training_conf.dataset_save_dir, training_conf.dataset_save_subname))
