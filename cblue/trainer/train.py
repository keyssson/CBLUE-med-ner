"""CMeEE NER 的训练、验证和预测循环。"""

import json
import math
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from cblue.metrics import ner_metric, write_ner_predictions


MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids")


class NERTrainer:
    def __init__(
        self,
        args,
        model,
        tokenizer,
        processor,
        logger,
        train_dataset=None,
        eval_dataset=None,
    ):
        self.args = args
        self.model = model.to(args.device)
        self.tokenizer = tokenizer
        self.processor = processor
        self.logger = logger
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def _dataloader(self, dataset, batch_size, shuffle=False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def _model_inputs(self, batch, include_labels=False):
        inputs = {
            key: batch[key].to(self.args.device)
            for key in MODEL_INPUT_KEYS
            if key in batch
        }
        if include_labels:
            inputs["labels"] = batch["labels"].to(self.args.device)
        return inputs

    def train(self):
        if self.train_dataset is None or self.eval_dataset is None:
            raise ValueError("训练需要 train_dataset 和 eval_dataset")

        train_loader = self._dataloader(
            self.train_dataset, self.args.train_batch_size, shuffle=True
        )
        total_steps = len(train_loader) * self.args.epochs
        if total_steps == 0:
            raise ValueError("训练集为空")
        warmup_steps = math.floor(total_steps * self.args.warmup_proportion)

        no_decay = ("bias", "LayerNorm.weight")
        optimizer_groups = [
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if not any(item in name for item in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if any(item in name for item in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_groups,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.logger.info("***** Running training *****")
        self.logger.info("Num samples: %d", len(self.train_dataset))
        self.logger.info("Num epochs: %d", self.args.epochs)
        self.logger.info("Total steps: %d; warmup steps: %d", total_steps, warmup_steps)

        global_step = 0
        best_f1 = -1.0
        best_epoch = 0
        patience_count = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                outputs = self.model(**self._model_inputs(batch, include_labels=True))
                loss = outputs.loss
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                epoch_loss += loss.item()
                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    self.logger.info(
                        "Epoch %d/%d - step %d/%d - loss %.6f",
                        epoch,
                        self.args.epochs,
                        global_step,
                        total_steps,
                        loss.item(),
                    )

            self.logger.info(
                "Epoch %d average loss: %.6f", epoch, epoch_loss / len(train_loader)
            )
            metrics = self.evaluate()
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_epoch = epoch
                patience_count = 0
                self._save_best_model(metrics, epoch)
            else:
                patience_count += 1
                self.logger.info(
                    "Early stopping counter: %d/%d",
                    patience_count,
                    self.args.earlystop_patience,
                )
                if patience_count >= self.args.earlystop_patience:
                    break

        self.logger.info("Training finished; best epoch=%d, best F1=%.6f", best_epoch, best_f1)
        return {"best_epoch": best_epoch, "best_f1": best_f1, "global_step": global_step}

    def _decode_batch(self, logits, batch):
        token_predictions = logits.argmax(dim=-1).cpu()
        word_ids = batch["word_ids"]
        sample_indices = batch["sample_index"].tolist()
        char_lengths = batch["char_length"].tolist()
        decoded = []

        for row, sample_index in enumerate(sample_indices):
            char_predictions = [self.processor.label2id["O"]] * char_lengths[row]
            seen_word_ids = set()
            for token_index, word_id_tensor in enumerate(word_ids[row]):
                word_id = int(word_id_tensor)
                if word_id < 0 or word_id in seen_word_ids or word_id >= char_lengths[row]:
                    continue
                char_predictions[word_id] = int(token_predictions[row, token_index])
                seen_word_ids.add(word_id)
            decoded.append((sample_index, char_predictions))
        return decoded

    def evaluate(self):
        if self.eval_dataset is None:
            raise ValueError("验证需要 eval_dataset")
        eval_loader = self._dataloader(
            self.eval_dataset, self.args.eval_batch_size, shuffle=False
        )
        predictions = [None] * len(self.eval_dataset)
        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                outputs = self.model(**self._model_inputs(batch))
                for sample_index, labels in self._decode_batch(outputs.logits, batch):
                    predictions[sample_index] = labels

        gold_labels = [sample["label_ids"] for sample in self.eval_dataset.samples]
        precision, recall, f1 = ner_metric(predictions, gold_labels, self.processor.id2label)
        metrics = {"precision": precision, "recall": recall, "f1": f1}
        self.logger.info(
            "Evaluation - precision: %.6f - recall: %.6f - F1: %.6f",
            precision,
            recall,
            f1,
        )
        return metrics

    def predict(self, test_dataset):
        test_loader = self._dataloader(
            test_dataset, self.args.eval_batch_size, shuffle=False
        )
        label_predictions = [None] * len(test_dataset)
        self.model.eval()
        self.logger.info("***** Running prediction: %d samples *****", len(test_dataset))
        with torch.no_grad():
            for batch in test_loader:
                outputs = self.model(**self._model_inputs(batch))
                for sample_index, labels in self._decode_batch(outputs.logits, batch):
                    label_predictions[sample_index] = labels

        entity_predictions = [
            self.processor.decode(labels, text)
            for labels, text in zip(label_predictions, test_dataset.orig_texts)
        ]
        output_path = write_ner_predictions(
            test_dataset.orig_texts,
            entity_predictions,
            self.args.result_output_dir,
        )
        self.logger.info("Prediction complete: %s", output_path)
        return output_path

    def _save_best_model(self, metrics, epoch):
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

        training_args = {
            key: str(value) if key == "device" else value
            for key, value in vars(self.args).items()
        }
        with open(
            os.path.join(self.args.output_dir, "training_args.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(training_args, file, indent=2, ensure_ascii=False)
        with open(
            os.path.join(self.args.output_dir, "eval_metrics.json"), "w", encoding="utf-8"
        ) as file:
            json.dump({"epoch": epoch, **metrics}, file, indent=2, ensure_ascii=False)
        self.logger.info("Saved new best model to %s", self.args.output_dir)
