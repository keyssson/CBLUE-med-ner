#!/usr/bin/env python3
"""CMeEE 命名实体识别的训练、验证和推理入口。"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cblue.data.data_process import NERDataProcessor  # noqa: E402
from cblue.data.dataset import NERDataset  # noqa: E402
from cblue.trainer.train import NERTrainer  # noqa: E402
from cblue.utils import init_logger, seed_everything  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate/predict CMeEE NER")
    parser.add_argument("--data_dir", default="CBLUEDatasets", help="包含 CMeEE/ 的数据集根目录")
    parser.add_argument(
        "--pretrained_model",
        default="data/model_data/chinese-bert-wwm-ext",
        help="训练时使用的 Hugging Face 格式预训练模型目录或模型名",
    )
    parser.add_argument("--output_dir", default="data/output/cmeee_ner", help="最优模型输出目录")
    parser.add_argument("--result_output_dir", default="data/result_output", help="预测 JSON 输出目录")
    parser.add_argument("--do_train", action="store_true", help="训练，并在每个 epoch 后验证")
    parser.add_argument("--do_eval", action="store_true", help="使用 output_dir 中的模型验证")
    parser.add_argument("--do_predict", action="store_true", help="使用 output_dir 中的模型预测测试集")

    parser.add_argument("--max_length", type=int, default=128, help="分词后的最大序列长度")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--earlystop_patience", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto 会优先使用 CUDA",
    )
    args = parser.parse_args()

    if not (args.do_train or args.do_eval or args.do_predict):
        parser.error("至少指定 --do_train、--do_eval、--do_predict 中的一项")
    if args.max_length < 4:
        parser.error("--max_length 必须至少为 4")
    if args.epochs < 1:
        parser.error("--epochs 必须至少为 1")
    if args.earlystop_patience < 1:
        parser.error("--earlystop_patience 必须至少为 1")
    if not 0.0 <= args.warmup_proportion <= 1.0:
        parser.error("--warmup_proportion 必须在 [0, 1] 范围内")
    return args


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境未检测到可用 CUDA")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fast_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(
            "该流程需要 Fast Tokenizer 来对齐字符与 BIO 标签；"
            f"{model_path} 未能加载出 Fast Tokenizer。"
        )
    return tokenizer


def build_dataset(processor, tokenizer, split, max_length):
    return NERDataset(
        samples=processor.get_samples(split),
        tokenizer=tokenizer,
        max_length=max_length,
        no_entity_id=processor.label2id[processor.no_entity_label],
    )


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_output_dir, exist_ok=True)

    logger = init_logger(os.path.join(args.output_dir, "cmeee_ner.log"))
    logger.info("Device: %s", args.device)
    logger.info("Arguments: %s", args)
    seed_everything(args.seed)

    processor = NERDataProcessor(args.data_dir)

    if args.do_train:
        tokenizer = load_fast_tokenizer(args.pretrained_model)
        train_dataset = build_dataset(processor, tokenizer, "train", args.max_length)
        eval_dataset = build_dataset(processor, tokenizer, "dev", args.max_length)
        model = AutoModelForTokenClassification.from_pretrained(
            args.pretrained_model,
            num_labels=processor.num_labels,
            id2label=processor.id2label,
            label2id=processor.label2id,
        )
        trainer = NERTrainer(
            args=args,
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            logger=logger,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()

    if args.do_eval or args.do_predict:
        if not os.path.isfile(os.path.join(args.output_dir, "config.json")):
            raise FileNotFoundError(
                f"{args.output_dir} 中没有训练后模型；请先运行 --do_train，"
                "或把已有微调模型放到该目录。"
            )
        tokenizer = load_fast_tokenizer(args.output_dir)
        model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
        if model.config.num_labels != processor.num_labels:
            raise ValueError(
                f"模型标签数为 {model.config.num_labels}，CMeEE 流程要求 {processor.num_labels}。"
            )

        eval_dataset = None
        if args.do_eval:
            eval_dataset = build_dataset(processor, tokenizer, "dev", args.max_length)
        trainer = NERTrainer(
            args=args,
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            logger=logger,
            eval_dataset=eval_dataset,
        )
        if args.do_eval:
            trainer.evaluate()
        if args.do_predict:
            test_dataset = build_dataset(processor, tokenizer, "test", args.max_length)
            result_path = trainer.predict(test_dataset)
            logger.info("Prediction file: %s", result_path)


if __name__ == "__main__":
    main()
