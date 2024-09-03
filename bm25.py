from rank_bm25 import BM25Okapi
from typing import List
from datasets import Blank

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TaskSpecificBM25:
    def __init__(self, examples, args):
        # turn examples to dict. Structure is {task_id : example}
        self.example_dict = {example.task_id: example for example in examples}
        self.args = args
    
    def query(self, task_ids: List[int], queries: List[str], topk: int):
        # calculate topk cross_context' point
        results = []
        for task_id, query in zip(task_ids, queries):
            # get example from dict
            example = self.example_dict.get(task_id)
            if example:
                # get crossfile_context's text. because the crossfile_context is a dict, structure is {id:xxx,title:xxx,text:xxx}
                candidate_context = example.crossfile_context

                # get text and split
                candidate_texts = [doc['text'].lower().split() for doc in candidate_context]
                # calculate every text's bm25 point
                query_tokens = query.lower().split()
                bm25 = BM25Okapi(candidate_texts)
                scores = bm25.get_scores(query_tokens)

                # get top-k text
                topk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]

                topk_context = [Blank("",candidate_context[i]['text']) for i in topk_indices]
                # the item of result is crossfile_context, which is a dict.
                results.append(topk_context)
            else:
                results.append([])
        return results