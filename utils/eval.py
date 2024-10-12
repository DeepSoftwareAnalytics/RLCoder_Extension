import json
import re
import argparse
import string
from collections import Counter
from rouge_score import rouge_scorer
import mauve

def accuracy(preds, labels):
    '''
    :param preds:  pred of all question
    :param labels: answers of all question, and the item is a list about some words
    '''
    def match(prediction, ground_truth):
        for gt in ground_truth:
            if gt in prediction:
                return 1
        return 0
    
    match_count = 0
    for pred, label in zip(preds, labels):
        match_count += match(pred, label)

    return 1.0 * match_count / len(preds)

def compute_acc(output_dir, ground_truth_dir, arc=False):
    """
    计算准确率 (acc)，即预测答案和真实答案完全匹配的比例
    :param output_dir: 生成的预测结果文件 (JSONL)
    :param ground_truth_dir: 真实答案文件 (JSONL)
    :return: 包含准确率的结果字典
    """
    # 读取预测结果
    predictions = []
    with open(f"{output_dir}/prediction.jsonl", 'r', encoding='utf-8') as pred_file:
        for line in pred_file:
            data = json.loads(line)
            predictions.append(data['pred'].strip())  # 收集模型预测的答案

    # 读取真实答案
    if arc == False :
        ground_truths = []
        with open(ground_truth_dir, 'r', encoding='utf-8') as gt_file:
            for line in gt_file:
                data = json.loads(line)
                ground_truths.append(data['answers'])  # 将真实答案作为列表传入
    else :
        ground_truths = []
        with open(ground_truth_dir, 'r', encoding='utf-8') as gt_file:
            for line in gt_file:
                data = json.loads(line)
                ground_truths.append(data['answerKey'])  # 将真实答案作为列表传入

    # 调用 accuracy 方法
    acc = accuracy(predictions, ground_truths)
    # 返回结果字典
    with open(f"{output_dir}/results.json", 'w') as f:
        res = {
            "acc": acc
    }
        f.write(json.dumps(res, indent=2))
    return acc


def compute_ASQA(output_dir, ground_truth_dir, arc=False):
    """
    计算针对ASQA bench的五个metric，包括em(exact match)、rg(rouge)、mau(mauve)、pre(citation precision)、rec(recall)
    :param output_dir: 生成的预测结果文件 (JSONL)
    :param ground_truth_dir: 真实答案文件 (JSONL)
    :return: 包含准确率的结果字典
    """
    # 读取预测结果
    predictions = []
    with open(f"{output_dir}/prediction.jsonl", 'r', encoding='utf-8') as pred_file:
        for line in pred_file:
            data = json.loads(line)
            predictions.append(data['pred'].strip())  # 收集模型预测的答案 (不需要分词，在标准化时会进行)

    # 读取真实答案
    ground_truths = []
    with open(ground_truth_dir, 'r', encoding='utf-8') as gt_file:
        for line in gt_file:
            data = json.loads(line)
            ground_truths.append(data['answer'])  # 将真实答案作为列表传入


    # calculate metric
    em = EM(predictions, ground_truths)
    mau = MAU(predictions, ground_truths)
    pre, rec, rg = RG(predictions, ground_truths)
    # 返回结果字典
    with open(f"{output_dir}/results.json", 'w') as f:
        res = {
            "em": em, 
            "rg": rg,
            "mau": mau,
            "pre": pre,
            "rec": rec
    }
        f.write(json.dumps(res, indent=2))
    return em,rg,mau,pre,rec 

def normalize(str):
    """
    将文本进行标准化
    :param str: 表示prediction或groundtruth的文本
    :return: 标准化的结果
    """
    # 去除冠词
    # 对于EM需要去除，对于ROUGE不需要
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    # 根据空格分词
    def white_space_fix(text):
        return ' '.join(text.split())
    # 去除标点符号
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    # 将全部字母小写
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str))))
    

def EM(predictions, ground_truths):

    def exact_match(prediction, ground_truth):
        return prediction == ground_truth
    
    # normalize 
    predictions_norm = [normalize(str) for str in predictions]
    ground_truths_norm = [normalize(str) for str in ground_truths]
    match_count = 0
    for prediction, ground_truth in zip(predictions_norm, ground_truths_norm):
        match_count += exact_match(prediction, ground_truth)

    return 1.0 * match_count / len(predictions)


def RG(predictions, ground_truths):

    def rouge_L(prediction, ground_truth):
        # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # In this function, we calculation rouge score (F1,precision,recall) about ROUGE-1
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        # score(generated_text, reference_text)
        scores = scorer.score(ground_truth, prediction)
        return scores['rouge1'].precision, scores['rouge1'].recall, scores['rouge1'].fmeasure
    
    # normalize 
    predictions_norm = [normalize(str) for str in predictions]
    ground_truths_norm = [normalize(str) for str in ground_truths]
    p_count = 0
    r_count = 0
    f_count = 0
    for prediction, ground_truth in zip(predictions_norm, ground_truths_norm):
        p, r, f = rouge_L(prediction, ground_truth)  # Unpack the results
        p_count += p  # Add precision
        r_count += r  # Add recall
        f_count += f  # Add F1-score

    return 1.0 * p_count / len(predictions), 1.0 * r_count / len(predictions), 1.0 * f_count / len(predictions)


def MAU(predictions, ground_truths):
    # mauve uses gpt-2-large to eval the fluency about the sentence
    mauve_score = mauve.compute_mauve(p_text=predictions, q_text=ground_truths)
    return mauve_score.mauve
