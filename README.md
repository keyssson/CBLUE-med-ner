# CMeEE NER 最小基线

本仓库已收缩为只包含 CMeEE 中文医学命名实体识别（NER）的最小可运行链路，支持训练、开发集验证、测试集推理和结果格式检查。

完整的数据与模型目录、安装方法、终端指令、预期日志、输出 JSON 和字段含义见 [NER_PIPELINE.md](NER_PIPELINE.md)。

最短启动方式：

```bash
pip install -r requirements.txt
bash examples/run_ner.sh all
bash examples/run_ner.sh check
```
