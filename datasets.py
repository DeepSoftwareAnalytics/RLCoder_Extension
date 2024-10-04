import pandas as pd
import random
from sklearn.model_selection import train_test_split
from utils.prompt import TASK_INST_TRAIN, TASK_INST_EVAL
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
        data_frame = pd.read_json("eval_data/popqa_longtail_w_gs.jsonl", lines=True)
    if datasetname == 'arc':
        data_frame = pd.read_json("eval_data/arc_challenge_processed.jsonl", lines=True)
    if datasetname == 'pubhealth':
        data_frame = pd.read_json("eval_data/health_claims_processed.jsonl", lines=True)
    if datasetname == 'triviaqa':
        data_frame = pd.read_json("eval_data/triviaqa_test_w_gs.jsonl", lines=True)
    if datasetname == 'ASQA':
        data_frame = pd.read_json("eval_data/asqa_eval_gtr_top100.jsonl", lines=True)
    if datasetname == 'FactScore':
        data_frame = pd.read_json("eval_data/factscore_unlabeled_alpaca_13b_retrieval.jsonl", lines=True)
    
    if args.debug:
        data_frame = data_frame.sample(100)
    dataset = []
    if datasetname == 'popqa' or datasetname == 'triviaqa':
        for _,row in data_frame.iterrows():

            # create a new example object for each row
            dataset.append(
                Example(task_id=row['id'],              
                        question=row['question'],       
                        answer=row['answers'][0],       
                        crossfile_context=row['ctxs'])  
            )
    if datasetname == 'arc':
        instruction = "### Instruction:\n" + TASK_INST_EVAL[datasetname] + " ## Input:\n\n "
        for _,row in data_frame.iterrows():
            choices = row["choices"]
            result = ''.join(f" {label}:{text}" for label, text in zip(choices['label'], choices['text']))
            # question = instruction + row['question'] + result + "\n\n### The answer is:\n" 
            question = instruction + row['question'] + result
            # create a new example object for each row
            dataset.append(
                Example(task_id=row['id'],              
                        question=question,              
                        answer=row['answerKey'],        
                        crossfile_context=row['ctxs'])  
            )   

    if datasetname == 'pubhealth':
        instruction = "### Instruction:\n" + TASK_INST_EVAL[datasetname] + " ## Input:\n\n "
        for index,row in data_frame.iterrows():
            # question = instruction + row['question']  + "\n\n### The answer is:\n"
            question = instruction + row['question'] 
            # create a new example object for each row
            dataset.append(
                Example(task_id=f"pubhealth_{index}",              
                        question=question,              
                        answer=row['answers'][0],       
                        crossfile_context=row['ctxs'])  
            )

    if datasetname == 'ASQA':
        instruction = "### Instruction:\n" + TASK_INST_EVAL[datasetname] + " ## Input:\n\n "
        for index,row in data_frame.iterrows():
            # question = instruction + row['question'] + "\n\n### The answer is:\n"
            question = instruction + row['question']
            dataset.append(
                Example(task_id=f"ASQA_{index}",              
                        question=row['question'],              
                        answer=row['answer'],       
                        crossfile_context=row['docs'])  
            )
        
    if datasetname == 'FactScore':
        for index,row in data_frame.iterrows():
            dataset.append(
                Example(task_id=f"FactScore_{index}",             
                        question=row['question'],              
                        answer=row['answer'][0],       
                        crossfile_context=row['ctxs'])  
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
    data_frame = pd.read_json("data/alpaca.jsonl", lines=True)


    training_datasets, validation_datasets = train_test_split(
        data_frame, 
        test_size=validation_split, 
        random_state=random_seed, 
        shuffle=True  # sklearn's train_test_split shuffles by default if shuffle=True
    )

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
        entry = raw_data.iloc[data_index % len(raw_data)]
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