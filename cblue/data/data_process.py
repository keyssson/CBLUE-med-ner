"""CMeEE 数据读取、BIO 标注和实体解码。"""

import json
import os


ENTITY_TYPES = ("bod", "dep", "dis", "dru", "equ", "ite", "mic", "pro", "sym")


class NERDataProcessor:
    """把 CMeEE 的字符区间标注转换成单层 BIO 序列。"""

    def __init__(self, root, no_entity_label="O"):
        self.task_data_dir = root
        self.paths = {
            "train": os.path.join(self.task_data_dir, "CMeEE-V2_train.json"),
            "dev": os.path.join(self.task_data_dir, "CMeEE-V2_dev.json"),
            "test": os.path.join(self.task_data_dir, "CMeEE-V2_dev.json"),
        }
        self.no_entity_label = no_entity_label
        labels = [no_entity_label]
        for entity_type in ENTITY_TYPES:
            labels.extend((f"B-{entity_type}", f"I-{entity_type}"))
        self.id2label = dict(enumerate(labels))
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        self.num_labels = len(labels)

    def get_samples(self, split):
        if split not in self.paths:
            raise ValueError(f"未知数据切分: {split}")
        path = self.paths[split]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"缺少 {split} 数据文件: {path}")

        with open(path, "r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError(f"{path} 的顶层必须是 JSON 数组")

        samples = []
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{path} 第 {index} 条记录缺少字符串字段 text")
            text = record["text"]
            labels = [self.no_entity_label] * len(text)
            if split != "test":
                entities = record.get("entities")
                if not isinstance(entities, list):
                    raise ValueError(f"{path} 第 {index} 条记录缺少数组字段 entities")
                # 单层 BIO 无法同时表示嵌套实体；短实体先写、长实体后写，优先保留外层实体。
                def entity_sort_key(entity):
                    if not isinstance(entity, dict):
                        return (-1, -1)
                    start = entity.get("start_idx")
                    end = entity.get("end_idx")
                    if type(start) is not int or type(end) is not int:
                        return (-1, -1)
                    return (end - start, start)

                entities = sorted(entities, key=entity_sort_key)
                for entity in entities:
                    self._apply_entity(labels, text, entity, path, index)

            normalized_chars = ["，" if char.isspace() else char for char in text]
            samples.append(
                {
                    "text": text,
                    "chars": normalized_chars,
                    "label_ids": [self.label2id[label] for label in labels],
                }
            )
        return samples

    def _apply_entity(self, labels, text, entity, path, record_index):
        required = ("start_idx", "end_idx", "type")
        if not isinstance(entity, dict) or any(field not in entity for field in required):
            raise ValueError(f"{path} 第 {record_index} 条记录存在不完整实体: {entity}")
        start = entity["start_idx"]
        end = entity["end_idx"]
        entity_type = entity["type"]
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"实体下标必须是整数: {entity}")
        if start < 0 or end < start or end > len(text):
            raise ValueError(f"实体下标越界: {entity}; text 长度={len(text)}")
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"未知 CMeEE 实体类型 {entity_type!r}: {entity}")
        if "entity" in entity and entity["entity"] != text[start:end]:
            raise ValueError(
                f"实体文本与字符区间不一致: {entity}; "
                f"实际片段={text[start:end]!r}"
            )
        labels[start] = f"B-{entity_type}"
        for char_index in range(start + 1, end):
            labels[char_index] = f"I-{entity_type}"

    def decode(self, label_ids, text):
        """把逐字符标签恢复成 CMeEE 要求的实体列表。"""
        entities = []
        active = None

        def close_entity(end_index):
            if active is None:
                return
            start_index, entity_type = active
            entities.append(
                {
                    "start_idx": start_index,
                    "end_idx": end_index,
                    "type": entity_type,
                    "entity": text[start_index:end_index],
                }
            )

        for index, label_id in enumerate(label_ids[:len(text)]):
            label = self.id2label[int(label_id)]
            if label == self.no_entity_label:
                close_entity(index)
                active = None
                continue

            prefix, entity_type = label.split("-", 1)
            if prefix == "B" or active is None or active[1] != entity_type:
                close_entity(index)
                active = (index, entity_type)
        close_entity(min(len(label_ids), len(text)))
        return entities
