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