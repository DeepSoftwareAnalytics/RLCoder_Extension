import torch.nn as nn
import torch    
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import Blank
import numpy as np

def tokenize(text, tokenizer, max_length, is_query):
    """
    Converts text to a list of token ids.
    :param text: The text to be converted
    :param tokenizer: The tokenizer to use
    :param max_length: The maximum input length
    :param is_query: A flag indicating whether the text is a query
    :return: A list of token ids
    """
    tokens = tokenizer.tokenize(text)
    # special token: CLS SEP encoder-only
    extra_token = 4
    if is_query:
        tokens = tokens[-max_length - extra_token:]
    else:
        tokens = tokens[:max_length - extra_token]
    # construct the tokensequence
    tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + tokens + [tokenizer.sep_token]
    tokens_id = tokenizer.convert_tokens_to_ids(tokens)

    # 确保tokens_id的长度为max_length
    if len(tokens_id) > max_length:
        tokens_id = tokens_id[:max_length]
    padding_length = max_length - len(tokens_id)
    tokens_id += [tokenizer.pad_token_id] * padding_length

    return tokens_id


class CustomDataset(Dataset):
    """
    Custom dataset class for handling cross_context and queries.
    :param max_length: The maximum input length
    :param tokenizer: The tokenizer used
    :param examples: The samples in the dataset
    :param is_query: A flag indicating whether it is a query
    """
    def __init__(self, max_length, tokenizer, examples, query=False):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.examples = examples
        self.query = query

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = str(self.examples[idx])
        tokens_id = tokenize(text, self.tokenizer, self.max_length, self.query)
        return torch.tensor(tokens_id, dtype=torch.long)



class Retriever(nn.Module):
    """
    Retriever model, used to compute sentence embeddings and retrieve cross_context.
    :param args: A namespace containing configuration parameters
    """
    def __init__(self, args):
        super(Retriever, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_path)
        self.args = args
        if self.args.disable_retriever is False:
            self.model = AutoModel.from_pretrained(args.retriever_model_path)
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.eval()

    def forward(self, source_ids):
        """
        Forward propagation function, used to generate the embedding representation of the input.
        :param input_ids: The sequence of input IDs
        :return: The embedding representation
        """
        mask = source_ids.ne(self.tokenizer.pad_token_id)
        token_embeddings = self.model(source_ids, attention_mask=mask)[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def retrieve(self, queries, candidate_context, topk):
        """
        Retrieval function, used to retrieve the most relevant cross_context from a list of candidates for each query.
        :param queries: A list of queries
        :param candidate_context: A list of candidate cross_context
        :param topk: The number of top-k cross_context to return for each query
        :return: A list of top-k cross_context for each query
        """
        query_dataset = CustomDataset(self.args.retriever_query_context_length, self.tokenizer, queries, query=True)
        query_dataloader = DataLoader(query_dataset, batch_size=self.args.retriever_batch_size, shuffle=False, num_workers=self.args.num_workers)
        candidate_numbers = [len(x) for x in candidate_context]
        candidate_context = [x for y in candidate_context for x in y]
        context_dataset = CustomDataset(self.args.retriever_candidate_context_length, self.tokenizer, candidate_context, query=False)
        context_dataloader = DataLoader(context_dataset, batch_size=self.args.retriever_batch_size, shuffle=False, num_workers=self.args.num_workers)
        # context_dataloader = tqdm(context_dataloader, desc="Encoding context") if self.args.enable_tqdm else context_dataloader

        query_embeddings = []
        context_embeddings = []
        with torch.no_grad():
            for batch in query_dataloader:
                batch = batch.cuda()
                query_embeddings.append(self.forward(batch))
            for batch in context_dataloader:
                batch = batch.cuda()
                context_embeddings.append(self.forward(batch))
        query_embeddings = torch.cat(query_embeddings, dim=0)
        context_embeddings = torch.cat(context_embeddings, dim=0)

        scores = torch.mm(query_embeddings, context_embeddings.t())
        scores = scores.cpu().numpy()

        topk_codeblocks = []  # Stores top-k codeblocks for each query
        start_idx = 0
        for i, num_candidates in enumerate(candidate_numbers):
            if num_candidates == 0:
                topk_codeblocks.append([])  # If there are no candidates for this query, add an empty list
                continue
            query_scores = scores[i][start_idx:start_idx+num_candidates]  # Get scores for the current query
            topk_indices_query = query_scores.argsort()[-topk:][::-1]  # Get indices of top-k codeblocks
            topk_candidates_query = [candidate_context[start_idx + idx] for idx in topk_indices_query]

            if len(topk_candidates_query) < topk:
                topk_candidates_query += [Blank("Don't need cross context to completion", "")] * (topk - len(topk_candidates_query))
            topk_codeblocks.append(topk_candidates_query)
            start_idx += num_candidates

        return topk_codeblocks

