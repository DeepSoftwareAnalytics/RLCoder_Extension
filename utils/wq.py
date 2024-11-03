import json

def compute_wq(output_dir, ground_truth_dir):
    evaluate_prediction_file(prediction_path=output_dir, gold_path=ground_truth_dir)

def evaluate_prediction_file(prediction_path: str, gold_path: str):

    # predicted_answers = json.load(open(prediction_path, encoding="utf-8"))
    predicted_answers = {}
    with open(f"{prediction_path}/prediction.jsonl", encoding="utf-8") as pred_file:
        predictions = json.load(pred_file)
        for item in predictions:
            predicted_answers.append(item['pred'])
    groundtruth_answers = {}
    with open(gold_path, encoding="utf-8") as groundtruth_file:
        groundtruth = json.load(pred_file)
        for item in groundtruth:
            groundtruth_answers.append(item['answer'])

    acc,correct = evaluate(predicted_answers,groundtruth_answers)

    # Output predictions to file if an output path is given
    
    output_dict = {"em": acc, "correct": correct, "total": len(groundtruth_answers)}

    with open(f"{prediction_path}/results.json", "w", encoding="utf8") as outfile:
        json.dump(output_dict, outfile)

    return acc

def evaluate(predicted_answers, groundtruth_answers):
    correct_count = 0
    total_count = len(groundtruth_answers)

    for gt_answer in groundtruth_answers:
        if gt_answer in predicted_answers:
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy, correct_count