import os
import time
import json
import torch
import random
import argparse
import numpy as np

from generator import Generator
from bm25 import TaskSpecificBM25
from retriever import Retriever, tokenize
from datasets import load_test_dataset, load_train_and_valid_dataset, construct_dataset, Blank
from utils.eval import compute_acc

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from prettytable import PrettyTable
import copy

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set seed
def set_random_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_random_seed()


# Retrieves context based on different inference types.
def retrieve_context(args, examples, bm25, retriever, dataset_name, is_training=False, inference_type=None):
    """
    Retrieves context based on different inference types.
    :param args: An argument object containing configuration parameters.
    :param examples: Examples used for retrieval.
    :param bm25: An instance of the BM25 model.
    :param retriever: An instance of the retriever.
    :param dataset_name: The name of the dataset.
    :param is_training: Whether it is in training mode.
    :return: A list of retrieved context.
    """
    if inference_type is None:
        inference_type = args.inference_type
    if inference_type == "baseline":
        return None, [[] for _ in range(len(examples))]

    bm25_topk, unixcoder_topk = 5, 5
    if inference_type in ["bm25", "unixcoder", "unixcoder_with_rl"]:
        if dataset_name not in bm25:
            bm25[dataset_name] = TaskSpecificBM25(examples, args)

        if inference_type == "unixcoder":
            bm25_topk = 50 
        elif inference_type == "unixcoder_with_rl":
            # bm25_topk = args.sample_number * 10 
            # eval's cross_context numb are 25
            # and if candidate_context numb is big, token will overflow
            bm25_topk = 15
            unixcoder_topk = args.sample_number 

        queries = [example.question for example in examples]
        candidate_context = bm25[dataset_name].query([x.task_id for x in examples], queries, topk=bm25_topk)

        if args.enable_repocoder and inference_type == 'unixcoder_with_rl':
            _, retrieved_context = retrieve_context(args, examples, bm25, retriever_RLCoder, dataset_name, inference_type="unixcoder") 
            generations = generator.generate(examples, retrieved_context, args.generator_max_generation_length)

            queries = [query + '\n' + prediction for query, prediction in zip(queries, generations)]

        if inference_type == "bm25":
            return queries, candidate_context
        elif inference_type == "unixcoder":
            return queries, retriever.retrieve(queries, candidate_context, topk=unixcoder_topk)
        elif inference_type == "unixcoder_with_rl":
            if is_training:
                if args.disable_stop_block:
                    candidate_context = retriever.retrieve(queries, candidate_context, topk=unixcoder_topk)
                else:
                    candidate_context = retriever.retrieve(queries, candidate_context, topk=unixcoder_topk-1)
                    candidate_context = [x + [Blank("Don't need cross context for completion", "")] for x in candidate_context]
            else:
                if not args.disable_stop_block:
                    candidate_context = [x + [Blank("Don't need cross context for completion", "")] for x in candidate_context]
                candidate_context = retriever.retrieve(queries,  candidate_context, topk=unixcoder_topk)
            return queries, candidate_context

    raise ValueError("Unsupported inference type: {}".format(args.inference_type))


class CustomDataset(Dataset):
    def __init__(self, max_query_length, max_candidate_length, tokenizer, queries, candidates, labels):
        self.max_query_length = max_query_length
        self.max_candidate_length = max_candidate_length
        self.tokenizer = tokenizer
        self.queries = queries
        self.candidates = candidates
        self.labels = labels

    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query_tokens_id = tokenize(self.queries[idx], self.tokenizer, self.max_query_length, True)
        candidate_tokens_id = [tokenize(str(x), self.tokenizer, self.max_candidate_length, False) for x in self.candidates[idx]]
        return torch.tensor(query_tokens_id, dtype=torch.long), torch.tensor(candidate_tokens_id, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def run(args):
    popqa_eval = load_test_dataset(args,"popqa")
    
    # training_raw_data, eval_raw_data = load_train_and_valid_dataset()
    # args.data_per_epoch = len(training_raw_data)
    # eval_all_examples = construct_dataset(eval_raw_data, 100 if args.debug else 1000)

    all_eval_examples = {
        # "alpaca_eval": eval_all_examples,
        "popqa_eval": popqa_eval,
        #"triviaqa_eval": triviaqa_eval,
        # "arc_eval": arc_eval,
        # "pubhealth_eval": pubhealth_eval,
        # "ASQA_eval": ASQA_eval,
        # "FactScore_eval": FactScore_eval,
    }


    global generator
    generator = Generator(args)
    retriever = Retriever(args)


    if args.enable_repocoder:
        args_RLCoder = copy.deepcopy(args)
        args_RLCoder.retriever_model_path = args.rlcoder_model_path
        global retriever_RLCoder
        retriever_RLCoder = Retriever(args_RLCoder)
    

    if not args.enable_forward_generation:
        args.forward_generation_times = 1
    else:
        if args.forward_generation_times is None:
            args.forward_generation_times = 4

    bm25 = {}
    
    if args.eval:
        table = PrettyTable()
        table.field_names = ["Method", "Dataset", "Total Samples", "Loss", "PPL", "EM", "ACC", "FS", "RG", "MAU", "PRE", "REC", "Time (sec)"]

        
        for name, examples in all_eval_examples.items():
            start_time = time.time()
            print("Evaluating on {} dataset".format(name))
            
            temp_examples = copy.deepcopy(examples)
            temp_generations = []
                
            for _ in range(args.forward_generation_times):
                _, retrieved_context = retrieve_context(args, temp_examples, bm25, retriever, name)
                losses = generator.evaluate(examples, retrieved_context)

                results = {"em": "-","acc": "-","fs": "-","rg": "-","mau": "-","pre": "-","rec": "-"}
                if args.enable_generation:
                    generations = generator.generate(temp_examples, retrieved_context, args.generator_max_generation_length)

                    if not temp_generations:
                        temp_generations = generations
                    else:
                        temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                    for i in range(len(temp_examples)):
                        temp_examples[i].question = examples[i].question + temp_generations[i]
                        
            if args.enable_generation:

                if not os.path.exists(f"{args.output_dir}/{name}"):
                    os.makedirs(f"{args.output_dir}/{name}", exist_ok=True)
                with open(f"{args.output_dir}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                    for example, temp_generation in zip(examples, temp_generations):
                        f_pred.write(json.dumps({"task_id": example.task_id, "pred": temp_generation}) + "\n")      
                if name == "popqa_eval":
                    results['acc'] = compute_acc(f"{args.output_dir}/{name}", "eval_data/popqa_longtail_w_gs.jsonl")
                if name == "arc_eval":
                    results['acc'] = compute_acc(f"{args.output_dir}/{name}", "eval_data/arc_challenge_processed.jsonl", True)
                if name == "pubhealth_eval":
                    results['acc'] = compute_acc(f"{args.output_dir}/{name}", "eval_data/health_claims_processed.jsonl")
                if name == "triviaqa_eval":    
                    results['acc'] = compute_acc(f"{args.output_dir}/{name}", "eval_data/triviaqa_test_w_gs.jsonl")
                
            table.add_row(['raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["acc"], results["fs"], results["rg"], results["mau"], results["pre"],results["rec"], round(time.time() - start_time, 1)])

            print(table)
        
    else:
        print("data_per_epoch:{}, batch_size:{}, sample_number:{}, epoch:{}, inner_epoch:{}, lr:{}".format(args.data_per_epoch, args.batch_size,args.sample_number,args.epoch,args.inner_epoch,args.lr))
        optimizer = AdamW(retriever.model.parameters(), lr=args.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch * 0.2, num_training_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch)
    
        evaluate_table = {}
        for name, examples in all_eval_examples.items():
            evaluate_table[name] = PrettyTable()
            if 'codereval' in name:
                evaluate_table[name].field_names = ["Epoch", "Method", "Dataset", "Total Samples", "Loss", "PPL", "count", "all", "self", "slib", "plib", "class", "file", "project", "Time (sec)"]
            else:
                evaluate_table[name].field_names = ["Epoch", "Method", "Dataset", "Total Samples", "Loss", "PPL", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]

        training_table = PrettyTable()
        training_table.field_names = ["Epoch", "Dataset", "Total Samples", "Rewards", "Training Loss", "Time (sec)"]


        # retriever.model.eval()
        # for name, examples in all_eval_examples.items():
            
        #     start_time = time.time()
        #     temp_examples = copy.deepcopy(examples)
        #     temp_generations = []

                
        #     for _ in range(args.forward_generation_times):
        #         _, retrieved_context = retrieve_context(args, temp_examples, bm25, retriever, name) 
        #         losses = generator.evaluate(examples, retrieved_context)


        #         results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
        #         if args.enable_generation:
        #             generations = generator.generate(temp_examples, retrieved_context, args.generator_max_generation_length)

        #             if not temp_generations:
        #                 temp_generations = generations
        #             else:
        #                 temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
        #             for i in range(len(temp_examples)):
        #                 temp_examples[i].question = examples[i].question + temp_generations[i]
                        
        #     if args.enable_generation:

        #         if os.path.exists(f"{args.output_dir}/result_init/{name}") is False:
        #             os.makedirs(f"{args.output_dir}/result_init/{name}", exist_ok=True)
        #         with open(f"{args.output_dir}/result_init/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
        #             for example, temp_generation in zip(examples, temp_generations):
        #                 f_pred.write(json.dumps({"task_id": example.task_id, "pred": temp_generation}) + "\n")
              

        #     if 'codereval' in name:
        #         evaluate_table[name].add_row(["init", 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
        #     else:
        #         evaluate_table[name].add_row(["init", 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

        #     print(evaluate_table[name])


        for epoch in range(args.epoch):
            print("=" * 40 + "Epoch:{}".format(epoch) + "=" * 40)
            retriever.model.eval()
            start_time = time.time()
            results = {}
            results["Epoch"] = epoch


            training_examples = construct_dataset(training_raw_data, 100 if args.debug else args.data_per_epoch)

            #!!!
            queries, retrieved_context = retrieve_context(args, training_examples, bm25, retriever, "alpaca_training_{}".format(epoch), True)

            training_examples_dup = [x for x in training_examples for _ in range(args.sample_number)]
            training_context_dup = [[x] for y in retrieved_context for x in y]
            assert len(training_examples_dup) == len(training_context_dup)
            losses = generator.evaluate(training_examples_dup, training_context_dup)
            labels = torch.tensor([x for x in losses]).view(-1, args.sample_number).argmin(-1)
            results["Total Samples"] = len(queries)
            results["Rewards"] = labels.float().mean().item()

            retriever.model.train()
            total_loss = 0
            dataset = CustomDataset(args.retriever_query_context_length, args.retriever_candidate_context_length, retriever.tokenizer, queries, retrieved_context, labels.tolist())
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

            for inner_epoch in range(args.inner_epoch):
                for batch in dataloader:
                    source_ids, doc_ids, labels = [x.cuda() for x in batch]
                    queries_embeddings = retriever(source_ids)
                    doc_texts_embeddings = retriever(doc_ids.view(-1, doc_ids.shape[-1])).view(source_ids.shape[0], args.sample_number, -1)
                    logits = torch.einsum("ab,acb->ac", queries_embeddings, doc_texts_embeddings)*20
                    loss = torch.nn.CrossEntropyLoss()(logits, labels)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(retriever.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    total_loss += loss.item()

                if args.enable_sft:
                    retriever.model.eval()
                    for name, examples in all_eval_examples.items():
                        
                        start_time = time.time()
                        temp_examples = copy.deepcopy(examples)
                        temp_generations = []

                            
                        for _ in range(args.forward_generation_times):
                            _, retrieved_context = retrieve_context(args, temp_examples, bm25, retriever, name) 
                            losses = generator.evaluate(examples, retrieved_context)

                            results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                            if args.enable_generation:
                                generations = generator.generate(temp_examples, retrieved_context, args.generator_max_generation_length)

                                if not temp_generations:
                                    temp_generations = generations
                                else:
                                    temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                                for i in range(len(temp_examples)):
                                    temp_examples[i].question = examples[i].question + temp_generations[i]
                                    
                        if args.enable_generation:
                            if os.path.exists(f"{args.output_dir}/result_{inner_epoch}/{name}") is False:
                                os.makedirs(f"{args.output_dir}/result_{inner_epoch}/{name}", exist_ok=True)
                            with open(f"{args.output_dir}/result_{inner_epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                                for example, generation in zip(examples, temp_generations):
                                    f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")

                        if 'codereval' in name:
                            evaluate_table[name].add_row([inner_epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
                        else:
                            evaluate_table[name].add_row([inner_epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

                        print(evaluate_table[name])
                    
                    retriever.model.module.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{inner_epoch}")
                    retriever.tokenizer.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{inner_epoch}")

            results["Training Loss"] = total_loss/len(dataloader)/args.inner_epoch
            results["Time (sec)"] = round(time.time() - start_time, 1)
            training_table.add_row([results["Epoch"], "QA_training_{}".format(epoch), results["Total Samples"], results["Rewards"], results["Training Loss"], results["Time (sec)"]])
            print(training_table)

            retriever.model.eval()
            for name, examples in all_eval_examples.items():
                
                start_time = time.time()
                temp_examples = copy.deepcopy(examples)
                temp_generations = []
                    
                for _ in range(args.forward_generation_times):
                    _, retrieved_context = retrieve_context(args, temp_examples, bm25, retriever, name) 
                    losses = generator.evaluate(examples, retrieved_context)

                    results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                    if args.enable_generation:
                        generations = generator.generate(temp_examples, retrieved_context, args.generator_max_generation_length)

                        if not temp_generations:
                            temp_generations = generations
                        else:
                            temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                        for i in range(len(temp_examples)):
                            temp_examples[i].question = examples[i].question + temp_generations[i]
                            
                if args.enable_generation:
                    if os.path.exists(f"{args.output_dir}/result_{epoch}/{name}") is False:
                        os.makedirs(f"{args.output_dir}/result_{epoch}/{name}", exist_ok=True)
                    with open(f"{args.output_dir}/result_{epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                        for example, generation in zip(examples, temp_generations):
                            f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")

                if 'codereval' in name:
                    evaluate_table[name].add_row([epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
                else:
                    evaluate_table[name].add_row([epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

                print(evaluate_table[name])

            retriever.model.module.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{epoch}")
            retriever.tokenizer.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{epoch}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_model_path", default="deepseek-ai/deepseek-coder-1.3b-base", type=str, help="Generator model path")
    parser.add_argument("--generator_batch_size_per_gpu", default=32, type=int, help="Generator batch size per GPU")
    parser.add_argument("--generator_max_crossfile_length", default=512, type=int, help="Maximum cross-file length for the generator")
    parser.add_argument("--generator_max_context_length", default=1024, type=int, help="Maximum context length for the generator")
    # parser.add_argument("--generator_max_generation_length", default=64, type=int, help="Maximum generation length for the generator")
    parser.add_argument("--generator_max_generation_length", default=100, type=int, help="Maximum generation length for the generator")

    parser.add_argument("--disable_generator", action="store_true", help="Disable the generator")

    # parser.add_argument("--retriever_model_path", default="microsoft/unixcoder-base", type=str, help="Retriever model path")
    parser.add_argument("--retriever_model_path", default="facebook/contriever-msmarco", type=str, help="Retriever model path")
    # parser.add_argument("--retriever_batch_size_per_gpu", default=64, type=int, help="Retriever batch size per GPU")
    parser.add_argument("--retriever_batch_size_per_gpu", default=16, type=int, help="Retriever batch size per GPU")
    parser.add_argument("--disable_retriever", action="store_true", help="Disable the retriever")
    parser.add_argument("--retriever_query_context_length", default=256, type=int, help="Retriever query context length")
    parser.add_argument("--retriever_candidate_context_length", default=512, type=int, help="Retriever candidate context length")
    # parser.add_argument("--retriever_query_context_length", default=128, type=int, help="Retriever query context length")
    # parser.add_argument("--retriever_candidate_context_length", default=256, type=int, help="Retriever candidate context length")

    parser.add_argument("--inference_type", default="baseline", type=str, help="Inference type")
    parser.add_argument("--output_dir", default="results/baseline", type=str, help="Output directory")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation")
    parser.add_argument("--enable_tqdm", action="store_true", help="Enable progress bar")
    parser.add_argument("--enable_generation", action="store_true", help="Enable generation")
    parser.add_argument("--debug", action="store_true", help="Debug mode, use a small dataset")

    parser.add_argument("--num_workers", default=14, type=int, help="Number of CPU cores")
    parser.add_argument("--enable_fixed_block", action="store_true", help="Use fixed length blocks when building candidates")
    parser.add_argument("--enable_sft", action="store_true", help="Train using supervised learning methods")
    parser.add_argument("--disable_stop_block", action="store_true", help="Disable the stop block")

    parser.add_argument("--enable_repocoder", action="store_true", help="Use the repocoder method during generation")
    parser.add_argument("--rlcoder_model_path", default="", type=str, help="RLCoder model path")

    parser.add_argument("--do_codereval", action="store_true", help="Execute codereval evaluation in docker")
    parser.add_argument("--enable_forward_generation", action="store_true", help="Use progressive generation methods during inference")
    parser.add_argument("--forward_generation_times", default=4, type=int, help="Number of times for progressive generation")

    parser.add_argument("--epoch", default=20, type=int, help="Number of training epochs")
    parser.add_argument("--inner_epoch", default=1, type=int, help="Number of inner training epochs")
    # parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    # parser.add_argument("--sample_number", default=10, type=int, help="Number of samples")
    parser.add_argument("--sample_number", default=5, type=int, help="Number of samples")
    parser.add_argument("--data_per_epoch", default=2000, type=int, help="Amount of data per epoch")
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")


    print("Number of GPUs:", torch.cuda.device_count())

    args = parser.parse_args()
    args.generator_batch_size = args.generator_batch_size_per_gpu * torch.cuda.device_count()
    args.retriever_batch_size = args.retriever_batch_size_per_gpu * torch.cuda.device_count()

    run(args)
