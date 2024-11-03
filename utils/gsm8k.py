import json

def compute_gsm8k(output_dir, ground_truth_dir):
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

    em,wrong = evaluate(predicted_answers,groundtruth_answers)

    # Output predictions to file if an output path is given
    
    output_dict = {"em": em, "wrong fomat": wrong}

    with open(f"{prediction_path}/results.json", "w", encoding="utf8") as outfile:
        json.dump(output_dict, outfile)

    return em

def evaluate(predicted_answers, groundtruth_answers):
    correct_count = 0
    wrong_format = 0

    for predicted, ground_truth in zip(predicted_answers, groundtruth_answers):
        # Extract the ground truth answer
        ground_truth_idx = ground_truth.index('####')
        ground_truth = ground_truth[ground_truth_idx + len('####'):].strip()

        # Extract the predicted answer
        response = predicted.lower()
        if 'the answer is' not in response:
            predicted_answer = ''
            wrong_format += 1
        else:
            predicted_idx = response.index('the answer is')
            predicted_answer = response[predicted_idx + len('the answer is'):].strip()

        # Compare the ground truth and predicted answers
        if ground_truth in predicted_answer:
            correct_count += 1

    accuracy = correct_count / len(predicted_answers)
    return accuracy, wrong_format

