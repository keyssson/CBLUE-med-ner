"""NER 实体级评估指标。"""


def bio_to_spans(label_ids, id2label):
    """将 BIO 标签序列转换为 (start, end, type) 集合，end 为闭区间。"""
    spans = set()
    active = None

    def close_span(end_index):
        if active is not None:
            spans.add((active[0], end_index, active[1]))

    for index, label_id in enumerate(label_ids):
        label = id2label[int(label_id)]
        if label == "O":
            close_span(index - 1)
            active = None
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or active is None or active[1] != entity_type:
            close_span(index - 1)
            active = (index, entity_type)
    close_span(len(label_ids) - 1)
    return spans


def ner_metric(predictions, labels, id2label):
    """按实体边界和类型完全匹配计算 micro precision/recall/F1。"""
    if len(predictions) != len(labels):
        raise ValueError("预测样本数与标签样本数不一致")

    true_positive = 0
    predicted_total = 0
    gold_total = 0
    for predicted_labels, gold_labels in zip(predictions, labels):
        predicted_spans = bio_to_spans(predicted_labels, id2label)
        gold_spans = bio_to_spans(gold_labels, id2label)
        true_positive += len(predicted_spans & gold_spans)
        predicted_total += len(predicted_spans)
        gold_total += len(gold_spans)

    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
