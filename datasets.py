import pandas as pd
import random
from sklearn.model_selection import train_test_split
class Blank(object):
    def __init__(self, description, content):
        """
        Represents a blank object. It is two situation: 1.crosscontext  2.blank object for signal
        :param description: The description of object.
        :param content: The content of object.
        """
        self.description = description
        self.content = content
    
    def __str__(self):
        return "\n" + self.description + "\n" + self.content
    
class Example(object):
    def __init__(self, task_id, question, answer, crossfile_context):
        """
        Represents an example used for constructing a dataset.
        :param task_id: Task ID.
        :param question: The question from a pair of qa.
        :param answer: The answer from a pair of qa.
        :param crossfile_context: Relative context of the question.
        (The item of crossfile_context is a dict, and it's structure is {id: xxx,title:xxx, text:xxx})
        """
        self.task_id = task_id
        self.question = question
        self.answer = answer
        self.crossfile_context = crossfile_context
    
    def __str__(self):
        return (
            f"[Example]:\n"
            f"[Task ID]:\n{self.task_id}\n"
            f"[Question]:\n{self.question}\n"
            f"[Answer]:\n{self.answer}\n"
        )

def load_test_dataset(args, datasetname):
    """
    Loads a dataset for evaluation.
    :param args: Parameters containing various configurations.
    :param datasetname: The name of the dataset to load.
    :return: The loaded dataset.
    """
    if datasetname == 'popqa':
        data_frame = pd.read_json("~/wsq/eval_data/popqa_longtail_w_gs.jsonl")
    if datasetname == 'arc':
        data_frame = pd.read_json("~/wsq/eval_data/arc_challenge_processed.jsonl")
    if datasetname == 'pubhealth':
        data_frame = pd.read_json("~/wsq/eval_data/health_claims_processed.jsonl")
    if datasetname == 'ASQA':
        data_frame = pd.read_json("~/wsq/eval_data/asqa_eval_gtr_top100.json")
    if datasetname == 'FactScore':
        data_frame = pd.read_json("~/wsq/eval_data/factscore_unlabeled_alpaca_13b_retrieval.jsonl")
    


    if args.debug:
        data_frame = data_frame.sample(100)
    dataset = []
    for _,row in data_frame.iterrows:

        # create a new example object for each row
        dataset.append(
            Example(task_id=row['id'],              # eval数据集中有id
                    question=row['question'],       # left_context即为question
                    answer=row['answers'],          # 将ground_truth作为answer
                    crossfile_context=row['ctxs'])  # eval中已经过初筛
        )
    return dataset


def load_train_and_valid_dataset(validation_split=0.2, random_seed=42):
    """
    Loads the training dataset, and uses sklearn for split the dataset, 80% for train and 20% for valid.
    :return: The training dataset.
    """
    training_datasets = []
    validation_datasets = []

    # Load the data
    data_frame = pd.read_json("~/wsq/data_after/alpaca.json")

    # Convert DataFrame to a list of records (each record is a dict)
    data_records = data_frame.to_dict(orient='records')
    
    # Shuffle the data
    random.seed(random_seed)
    random.shuffle(data_records)
    
    # Split the data into training and validation sets
    training_datasets, validation_datasets = train_test_split(data_records, test_size=validation_split, random_state=random_seed)


    return training_datasets, validation_datasets

def construct_dataset(raw_data, num_samples):
    """
    Builds a dataset.
    :param raw_data: Raw data.
    :param num_samples: The number of samples to generate, the default is length of datasets.
    :return: The list of constructed samples.
    """

    examples = []
    data_index = 0
    
    while len(examples) < num_samples:
        # get an item from dataset
        entry = raw_data[data_index % len(raw_data)]
        data_index += 1

        # get data from columns
        task_id = entry['task_id']
        question = entry['left_context']
        answer = entry['groundtruth']
        crossfile_context = entry['crossfile_context']
        
        # create a new example object
        examples.append(
            Example(task_id=task_id,                        # 数据集中有task_id
                    question=question,                      # left_context即为question
                    answer=answer,                          # 将ground_truth作为answer
                    crossfile_context=crossfile_context)    #初筛后的top100个文档
        )

    return examples