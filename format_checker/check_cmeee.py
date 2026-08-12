#!/usr/bin/env python3
"""检查 CMeEE 测试集预测文件的结构、数量、实体类型和字符区间。"""

import argparse
import json
import os
import sys


ENTITY_TYPES = {"dis", "sym", "dru", "equ", "pro", "bod", "ite", "mic", "dep"}


def load_records(path):
    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"{path}: 顶层必须是 JSON 数组")
    return records


def check(source_path, submission_path):
    if os.path.basename(submission_path) != "CMeEE_test.json":
        raise ValueError("预测文件名必须是 CMeEE_test.json")

    source_records = load_records(source_path)
    submission_records = load_records(submission_path)
    if len(source_records) != len(submission_records):
        raise ValueError(
            f"记录数不一致：测试集 {len(source_records)} 条，预测文件 {len(submission_records)} 条"
        )

    for record_index, (source, predicted) in enumerate(
        zip(source_records, submission_records)
    ):
        if not isinstance(source, dict) or not isinstance(source.get("text"), str):
            raise ValueError(f"测试集第 {record_index} 条缺少字符串字段 text")
        if not isinstance(predicted, dict) or predicted.get("text") != source["text"]:
            raise ValueError(f"预测文件第 {record_index} 条 text 与测试集不一致")
        entities = predicted.get("entities")
        if not isinstance(entities, list):
            raise ValueError(f"预测文件第 {record_index} 条 entities 必须是数组")

        text = predicted["text"]
        for entity_index, entity in enumerate(entities):
            location = f"第 {record_index} 条的第 {entity_index} 个实体"
            if not isinstance(entity, dict):
                raise ValueError(f"{location}必须是对象")
            for field in ("start_idx", "end_idx", "type"):
                if field not in entity:
                    raise ValueError(f"{location}缺少字段 {field}")
            start = entity["start_idx"]
            end = entity["end_idx"]
            if type(start) is not int or type(end) is not int:
                raise ValueError(f"{location}的 start_idx/end_idx 必须是整数")
            if start < 0 or end < start or end >= len(text):
                raise ValueError(f"{location}的字符区间 [{start}, {end}] 越界")
            if entity["type"] not in ENTITY_TYPES:
                raise ValueError(f"{location}包含未知类型 {entity['type']!r}")
            if "entity" in entity and entity["entity"] != text[start:end + 1]:
                raise ValueError(
                    f"{location}的 entity 与 text[{start}:{end + 1}] 不一致"
                )


def main():
    parser = argparse.ArgumentParser(description="Check CMeEE prediction format")
    parser.add_argument("source", help="原始 CMeEE_test.json")
    parser.add_argument("submission", help="模型生成的 CMeEE_test.json")
    args = parser.parse_args()
    try:
        check(args.source, args.submission)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Format Check Failed: {error}", file=sys.stderr)
        return 1
    print("Format Check Success!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
