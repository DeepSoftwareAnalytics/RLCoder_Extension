import json

def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1

    return 100 * (match_count / len(preds))

def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0

def compute_acc(output_dir, ground_truth_dir):
    """
    计算准确率 (acc)，即预测答案和真实答案完全匹配的比例
    :param output_dir: 生成的预测结果文件 (JSONL)
    :param ground_truth_dir: 真实答案文件 (JSONL)
    :return: 包含准确率的结果字典
    """
    # 读取预测结果
    predictions = []
    with open(output_dir, 'r', encoding='utf-8') as pred_file:
        for line in pred_file:
            data = json.loads(line)
            predictions.append(data['pred'].strip())  # 收集模型预测的答案

    # 读取真实答案
    ground_truths = []
    with open(ground_truth_dir, 'r', encoding='utf-8') as gt_file:
        for line in gt_file:
            data = json.loads(line)
            ground_truths.append([data['answer'].strip()])  # 将真实答案作为列表传入

    # 调用 metrics.py 中的 accuracy 方法
    acc = accuracy(predictions, ground_truths)

    # 返回结果字典
    with open(f"{output_dir}/results.json", 'w') as f:
        res = {
            "acc": acc
    }
    f.write(json.dumps(res, indent=2))
    return {
        "acc": acc
    }
