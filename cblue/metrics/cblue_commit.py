"""写出 CMeEE 预测结果。"""

import json
import os


def write_ner_predictions(texts, predictions, output_dir):
    if len(texts) != len(predictions):
        raise ValueError("原文数与预测结果数不一致")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "CMeEE_test.json")
    records = [
        {"text": text, "entities": entities}
        for text, entities in zip(texts, predictions)
    ]
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
    return output_path
