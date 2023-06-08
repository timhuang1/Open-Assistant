"""
   Datasets for LM objective pre-training aimed to prevent catastrophic forgetting during fine-tuning
"""
import os
import json
from pathlib import Path

from datasets import load_dataset
from model_training.custom_datasets.formatting import DatasetEntryLm
from torch.utils.data import Dataset


class RedPajama(Dataset):
    name = "red_pajama"

    def __init__(self, cache_dir: str | Path, mode: str = "sft", char_max_len: str = 9216) -> None:
        super().__init__()
        self.mode = mode
        assert mode in ("sft", "rm", "rl")
        self.char_max_len = char_max_len

        self.dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir=cache_dir)["train"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> DatasetEntryLm:
        dialogue = DatasetEntryLm(text=self.dataset[index]["text"][: self.char_max_len])
        return dialogue


class LocalLM(Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str | Path,
        cache_dir: str | Path,
        mode: str = "sft",
        char_max_len: str = 9216,
        **kwargs
    ) -> None:
        super().__init__()
        self.mode = mode
        assert mode in ("sft", "rm", "rl")
        self.char_max_len = char_max_len
        assert (input_file_path := kwargs.get("input_file_path")) is not None, "Loading LocalDialogue ds requires passing input_file_path"
        # self.dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir=cache_dir)["train"]
        self.text_col = kwargs.get("lm_text_col", "content")
        self.dataset = [json.loads(ln) for ln in open(os.path.join(data_dir, input_file_path)).readlines()]
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> DatasetEntryLm:
        dialogue = DatasetEntryLm(text=self.dataset[index][self.text_col][: self.char_max_len])
        return dialogue
