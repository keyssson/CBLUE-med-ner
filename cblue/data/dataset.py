"""CMeEE 字符级 NER 数据集。"""

import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128, no_entity_id=0, ignore_label=-100):
        if not tokenizer.is_fast:
            raise ValueError("NERDataset 需要 Fast Tokenizer 提供 word_ids 字符对齐信息")
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.no_entity_id = no_entity_id
        self.ignore_label = ignore_label
        self.orig_texts = [sample["text"] for sample in samples]

    def __getitem__(self, index):
        sample = self.samples[index]
        encoded = self.tokenizer(
            sample["chars"],
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )
        word_ids = encoded.word_ids()
        labels = [self.ignore_label] * len(word_ids)
        previous_word_id = None
        for token_index, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous_word_id:
                labels[token_index] = sample["label_ids"][word_id]
            previous_word_id = word_id

        item = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "word_ids": torch.tensor(
                [-1 if word_id is None else word_id for word_id in word_ids], dtype=torch.long
            ),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "char_length": torch.tensor(len(sample["chars"]), dtype=torch.long),
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = torch.tensor(encoded["token_type_ids"], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.samples)
