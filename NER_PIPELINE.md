# CMeEE NER 完整运行手册

本文说明如何从零放置数据和预训练模型，完成 CMeEE 命名实体识别的训练、验证、测试集推理与结果格式检查。

## 1. 当前仓库范围

仓库只保留 CMeEE NER 链路：

```text
CBLUE-main/
├── baselines/run_ner.py             # 训练、验证、推理统一入口
├── cblue/
│   ├── data/                        # CMeEE 读取、BIO 构造、字符对齐
│   ├── metrics/                     # 实体级 micro P/R/F1、结果写出
│   ├── trainer/                     # 训练、早停、验证、预测
│   └── utils.py                     # 日志和随机种子
├── examples/run_ner.sh              # 可直接执行的命令封装
├── format_checker/check_cmeee.py    # 预测结果格式检查
├── requirements.txt
└── NER_PIPELINE.md
```

原仓库中的关系抽取、疾病标准化、文本分类、句子相似度、问句匹配和 ZEN 专用代码均不在本最小链路内。

## 2. 任务和标签含义

CMeEE 输入一段医学文本，输出实体的起止字符下标及类别。九种类别如下：

| 类型 | 含义 |
| --- | --- |
| `dis` | 疾病 |
| `sym` | 临床表现/症状 |
| `dru` | 药物 |
| `equ` | 医疗设备 |
| `pro` | 医疗程序 |
| `bod` | 身体部位 |
| `ite` | 医学检验项目 |
| `mic` | 微生物 |
| `dep` | 科室 |

模型内部使用字符级 BIO 标签。例如“肺炎”是疾病时，对应 `B-dis I-dis`；非实体字符是 `O`。

## 3. 环境安装

建议使用 Python 3.8（本仓库整理时的基础环境为 Python 3.8.20）。在仓库根目录执行：

```bash
cd /mnt/data/wangkesong/CBLUE-main
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

在当前服务器上，默认的 `cblue` Conda 环境没有安装 PyTorch/Transformers；现有的 `medner` 环境已验证可以跑通本链路，可直接使用：

```bash
conda activate medner
cd /mnt/data/wangkesong/CBLUE-main
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

本次实际冒烟验证所用版本为 PyTorch `2.4.0+cu121`、Transformers `4.48.3`。如果该环境之后被修改或删除，再按上面的虚拟环境步骤安装即可。

`requirements.txt` 将 Transformers 和 PyTorch 约束在已支持的主版本范围内。若服务器需安装特定 CUDA 版本的 PyTorch，请先按服务器 CUDA 环境安装对应 PyTorch，再执行：

```bash
pip install "transformers>=4.30,<5"
```

确认环境：

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print('cuda=', torch.cuda.is_available())"
```

## 4. 放置数据集

参数 `--data_dir` 指向“包含 `CMeEE` 子目录”的数据集总目录。使用默认值时，把文件放成：

```text
CBLUEDatasets/
└── CMeEE/
    ├── CMeEE_train.json
    ├── CMeEE_dev.json
    └── CMeEE_test.json
```

训练集和开发集是 JSON 数组，每条数据至少含 `text` 和 `entities`：

```json
[
  {
    "text": "患者出现肺炎。",
    "entities": [
      {
        "start_idx": 4,
        "end_idx": 5,
        "type": "dis",
        "entity": "肺炎"
      }
    ]
  }
]
```

`start_idx` 和 `end_idx` 都是从 0 开始的字符下标，且 `end_idx` 是闭区间。上例中的实体文本等于 `text[4:6]`。

测试集可以只有文本，也可以带空实体数组：

```json
[
  {"text": "患者出现肺炎。"}
]
```

运行前可检查三个文件是否就位：

```bash
ls -lh CBLUEDatasets/CMeEE/CMeEE_{train,dev,test}.json
```

## 5. 放置预训练模型

训练入口接受 Hugging Face 格式且带 Fast Tokenizer 的 BERT 类 token-classification 兼容模型。默认路径为：

```text
data/model_data/chinese-bert-wwm-ext/
├── config.json
├── pytorch_model.bin
├── vocab.txt
├── tokenizer_config.json        # 某些模型可能没有
└── special_tokens_map.json      # 某些模型可能没有
```

也可以把 `--pretrained_model` 设为其他本地模型目录。若运行环境允许访问 Hugging Face，也可以传模型 ID；离线服务器建议提前下载并使用本地目录。

模型目录与训练后的模型目录不要混淆：

- `data/model_data/chinese-bert-wwm-ext`：未经本任务微调的基础模型，供 `--do_train` 使用。
- `data/output/cmeee_ner`：训练保存的最佳 CMeEE 模型，供 `--do_eval` 和 `--do_predict` 使用。

## 6. 一条命令跑完整链路

在仓库根目录执行：

```bash
bash examples/run_ner.sh all
```

该命令等价于：

```bash
python baselines/run_ner.py \
  --data_dir CBLUEDatasets \
  --pretrained_model data/model_data/chinese-bert-wwm-ext \
  --output_dir data/output/cmeee_ner \
  --result_output_dir data/result_output \
  --do_train \
  --do_predict \
  --max_length 128 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 3e-5 \
  --epochs 5 \
  --warmup_proportion 0.1 \
  --earlystop_patience 3 \
  --logging_steps 100 \
  --seed 2021
```

执行顺序是：

1. 读取训练集，按字符生成单层 BIO 标签。
2. Fast Tokenizer 分词，并通过 `word_ids` 把 token 与原字符对齐。
3. 微调 `AutoModelForTokenClassification`。
4. 每个 epoch 在开发集计算实体级 micro precision、recall、F1。
5. 保存 F1 最高的模型；连续 3 个 epoch 无提升则早停。
6. 重新加载最佳模型，对测试集推理。
7. 把 BIO 序列恢复为实体区间，生成 `data/result_output/CMeEE_test.json`。

## 7. 分阶段运行的具体指令

只训练（训练过程中仍会自动验证）：

```bash
bash examples/run_ner.sh train
```

只验证已经保存到 `data/output/cmeee_ner` 的模型：

```bash
bash examples/run_ner.sh eval
```

只做测试集推理：

```bash
bash examples/run_ner.sh predict
```

如果路径不是默认值，可用环境变量覆盖脚本配置：

```bash
DATA_DIR=/path/to/datasets \
PRETRAINED_MODEL=/path/to/pretrained-model \
OUTPUT_DIR=/path/to/finetuned-cmeee \
RESULT_OUTPUT_DIR=/path/to/results \
MAX_LENGTH=256 \
bash examples/run_ner.sh all
```

也可以直接调用 `baselines/run_ner.py`。查看全部参数：

```bash
python baselines/run_ner.py --help
```

强制 CPU 或 CUDA：

```bash
python baselines/run_ner.py ... --do_predict --device cpu
python baselines/run_ner.py ... --do_predict --device cuda
```

默认 `--device auto`，检测到 CUDA 时使用 GPU，否则使用 CPU。

## 8. 预计终端输出及含义

日志会同时打印到终端，并写入 `data/output/cmeee_ner/cmeee_ner.log`。典型输出如下：

```text
2026-08-07 14:00:00 - INFO - Device: cuda
2026-08-07 14:00:03 - INFO - ***** Running training *****
2026-08-07 14:00:03 - INFO - Num samples: 15000
2026-08-07 14:03:20 - INFO - Epoch 1 average loss: 0.214532
2026-08-07 14:03:41 - INFO - Evaluation - precision: 0.671200 - recall: 0.642100 - F1: 0.656328
2026-08-07 14:03:42 - INFO - Saved new best model to data/output/cmeee_ner
...
2026-08-07 14:15:20 - INFO - Training finished; best epoch=4, best F1=0.684321
2026-08-07 14:15:55 - INFO - Prediction complete: data/result_output/CMeEE_test.json
```

指标含义：

- `precision`：预测出的实体中，边界和类型都完全正确的比例。
- `recall`：金标准实体中，被模型以正确边界和类型找回的比例。
- `F1`：precision 和 recall 的调和平均；本流程用实体级 micro 统计。
- `loss`：训练优化目标，只用于观察收敛，不能直接当成任务准确率。

数值受预训练模型、随机种子、显卡计算、超参数和数据版本影响，示例数值只展示日志样式，不是承诺结果。

## 9. 模型和结果文件

训练完成后，`data/output/cmeee_ner/` 预计包含：

```text
config.json
pytorch_model.bin                 # 新版保存配置也可能写 model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.txt
training_args.json
eval_metrics.json
cmeee_ner.log
```

测试集预测写入 `data/result_output/CMeEE_test.json`，样式如下：

```json
[
  {
    "text": "患者出现肺炎。",
    "entities": [
      {
        "start_idx": 4,
        "end_idx": 5,
        "type": "dis",
        "entity": "肺炎"
      }
    ]
  }
]
```

字段含义：

- `text`：原测试文本，内容和顺序不变。
- `entities`：模型识别出的实体数组；没有实体时为 `[]`。
- `start_idx` / `end_idx`：实体在 `text` 中的开始/结束字符下标，均为闭区间。
- `type`：九种实体类别之一。
- `entity`：`text[start_idx:end_idx + 1]` 对应的文本片段，便于人工检查。

## 10. 检查预测格式

生成结果后执行：

```bash
bash examples/run_ner.sh check
```

等价的直接命令是：

```bash
python format_checker/check_cmeee.py \
  CBLUEDatasets/CMeEE/CMeEE_test.json \
  data/result_output/CMeEE_test.json
```

成功输出：

```text
Format Check Success!
```

检查项包括文件名、JSON 顶层结构、记录数与文本顺序、必需字段、下标范围、实体类别，以及 `entity` 和字符区间是否一致。格式通过只代表文件结构合法，不代表实体预测正确。

## 11. 长文本、嵌套实体和常见问题

### 长文本

`max_length` 包含模型特殊 token。文本分词后超过该长度会截断；截断范围外字符会被当作 `O`，因此应按数据长度和显存调整，例如：

```bash
MAX_LENGTH=256 bash examples/run_ner.sh all
```

提高长度会增加显存占用。显存不足时优先减小 `train_batch_size` 和 `eval_batch_size`。

### 嵌套实体

本仓库保留的是原基线思路的最小单层 BIO 流程。单层 BIO 不能同时表达相互嵌套的两个实体；数据预处理在重叠时优先保留较长的外层实体。因此开发集指标也是该单层标签空间上的指标，不等同于支持嵌套结构的模型上限。若业务要求完整保留嵌套实体，需要多头标注、span 分类或其他嵌套 NER 模型，这超出本最小基线范围。

### 常见报错

- `缺少 train/dev/test 数据文件`：确认 `--data_dir` 指向 `CMeEE/` 的上一级目录。
- `没有训练后模型`：先运行 `bash examples/run_ner.sh train`，或把微调后的模型放到 `--output_dir`。
- `需要 Fast Tokenizer`：模型目录必须能被 `AutoTokenizer(..., use_fast=True)` 加载，并生成 Fast Tokenizer。
- `CUDA out of memory`：减小 batch size 或 `MAX_LENGTH`，也可先用 `--device cpu` 验证流程。
- `实体下标越界`：训练/开发集中的 `start_idx`、`end_idx` 与文本字符位置不一致，需要修正数据。
- 推理很慢：确认日志中的 `Device` 是否为 `cuda`，并检查 PyTorch 是否安装了匹配的 CUDA 版本。
- 注意新的v2数据集`end_idx`指向的是实体的后一位，所以 [start_idx, end_idx] 即实体边界