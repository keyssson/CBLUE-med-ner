#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-CBLUEDatasets}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-data/model_data/chinese-bert-wwm-ext}"
OUTPUT_DIR="${OUTPUT_DIR:-data/output/cmeee_ner}"
RESULT_OUTPUT_DIR="${RESULT_OUTPUT_DIR:-data/result_output}"
MAX_LENGTH="${MAX_LENGTH:-128}"

MODE="${1:-all}"
COMMON_ARGS=(
  --data_dir "${DATA_DIR}"
  --pretrained_model "${PRETRAINED_MODEL}"
  --output_dir "${OUTPUT_DIR}"
  --result_output_dir "${RESULT_OUTPUT_DIR}"
  --max_length "${MAX_LENGTH}"
)

case "${MODE}" in
  all)
    python baselines/run_ner.py "${COMMON_ARGS[@]}" \
      --do_train --do_predict \
      --train_batch_size 16 \
      --eval_batch_size 32 \
      --learning_rate 3e-5 \
      --epochs 5 \
      --warmup_proportion 0.1 \
      --earlystop_patience 3 \
      --logging_steps 100 \
      --seed 2021
    ;;
  train)
    python baselines/run_ner.py "${COMMON_ARGS[@]}" \
      --do_train \
      --train_batch_size 16 \
      --eval_batch_size 32 \
      --learning_rate 3e-5 \
      --epochs 5 \
      --warmup_proportion 0.1 \
      --earlystop_patience 3 \
      --logging_steps 100 \
      --seed 2021
    ;;
  eval)
    python baselines/run_ner.py "${COMMON_ARGS[@]}" --do_eval --eval_batch_size 32
    ;;
  predict)
    python baselines/run_ner.py "${COMMON_ARGS[@]}" --do_predict --eval_batch_size 32
    ;;
  check)
    python format_checker/check_cmeee.py \
      "${DATA_DIR}/CMeEE/CMeEE_test.json" \
      "${RESULT_OUTPUT_DIR}/CMeEE_test.json"
    ;;
  *)
    echo "Usage: bash examples/run_ner.sh [all|train|eval|predict|check]" >&2
    exit 2
    ;;
esac
