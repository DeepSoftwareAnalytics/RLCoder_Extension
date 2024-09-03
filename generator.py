import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import re

class CustomDataset(Dataset):
    """
    A dataset class for generation.

    Args:
        args: Configuration parameters.
        tokenizer: Tokenizer.
        examples: A collection of examples.
        retrieved_context: Retrieved context.
    """
    def __init__(self, args, tokenizer, examples, retrieved_context, generation=False):
        self.args = args
        self.tokenizer = tokenizer
        self.examples = examples
        self.retrieved_context = retrieved_context
        self.generation = generation

    def __len__(self):
        return len(self.examples)

    def construct_prompts(self,example,retrieved_context):

        filter_blanks = []
        for x in retrieved_context:
            # get cross_context and drop blank
            if x.content != "":
                filter_blanks.append(x)
            else:
                break
        crossfile_context = "\n\n".join([str(context) for context in filter_blanks])

        # limit cross_contex's length
        crossfile_context = self.tokenizer.encode(crossfile_context[:self.args.generator_max_crossfile_length], add_special_tokens=False)
        
        # limit infile_context's length
        allowed_prompt_length = self.args.generator_max_context_length - len(crossfile_context)
        infile_context = self.tokenizer.encode(example.question, add_special_tokens=False)[-allowed_prompt_length:]

        # join prompt
        prompt = self.tokenizer.decode(crossfile_context + infile_context)
        return prompt


    def __getitem__(self, idx):
        example = self.examples[idx]
        retrieved_context = self.retrieved_context[idx]
        prompt = self.construct_prompts(example,retrieved_context)
        
        prompt_ids = self.tokenizer.encode(prompt)[-self.args.generator_max_context_length:]
        if self.generation:
             padding_length = self.args.generator_max_context_length - len(prompt_ids)
             input_ids = [self.tokenizer.pad_token_id] * padding_length + prompt_ids
             return torch.tensor(input_ids)

        target_ids = self.tokenizer.encode(example.answer, add_special_tokens=False)[:self.args.generator_max_generation_length]

        input_ids = prompt_ids + target_ids
        labels = [-100 for _ in prompt_ids] + target_ids

        padding_length = self.args.generator_max_context_length + self.args.generator_max_generation_length - len(input_ids)
        input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
        labels = [-100] * padding_length + labels 

        return torch.tensor(input_ids), torch.tensor(labels)


class Model(nn.Module):
    def __init__(self, generator_model_path, tokenizer, max_generation_length=64):
        super(Model, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(generator_model_path, torch_dtype=torch.float16)
        self.tokenizer = tokenizer
        self.max_generation_length = max_generation_length

    def forward(self, inputs=None, labels=None):
        """
        Forward propagation method for calculating loss.
        :param inputs: Input data.
        :param labels: Label data.
        :return: The average loss per sample.
        """
        if labels is not None:
            # Compute logits and calculate loss
            logits = self.base_model(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id))[0]
            logits = logits[:, :-1]  # Shift logits for next-token prediction
            labels = labels[:, 1:]   # Shift labels for next-token prediction
            
            # Calculate loss without applying any additional weights
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,  # Ignore padding tokens in the loss calculation
                reduction='none'
            )

            # Average the loss per label
            loss_per_label = loss.reshape(labels.size(0), -1).sum(dim=1) / labels.ne(-100).sum(dim=1)
            
            return loss_per_label
        else:
            # Generate text if no labels are provided
            generated_ids = self.base_model.generate(
                inputs, 
                attention_mask=inputs.ne(self.tokenizer.pad_token_id), 
                max_length=inputs.size(1) + self.max_generation_length, 
                pad_token_id=self.tokenizer.pad_token_id
            )
            return generated_ids[:, inputs.size(1):]
       

    
class Generator:
    """
    generator class.
    It is used to do two thing: 1.evaluate 2.generate
    Args:
        args: Configuration parameters.
    """
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path)
        self.tokenizer.model_max_length = 1e10
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if not args.disable_generator:
            self.model = Model(args.generator_model_path, self.tokenizer)
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.eval()

        self.args = args

    def evaluate(self, examples, retrieved_context):
        """
        Evaluates the generated results.

        Args:
            examples: A collection of examples.
            retrieved_context: Retrieved context.

        Returns:
            A list of loss values.
        """
        losses = []
        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_context)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)

        pbar = tqdm(dataloader, disable=not self.args.enable_tqdm)
        with torch.no_grad():
            for batch in pbar:
                inputs, labels = [x.cuda() for x in batch]
                loss_per_label = self.model(inputs, labels)
                losses.extend(loss_per_label.tolist())
                current_ppl = np.exp(np.mean(losses))
                pbar.set_description(f"Loss/PPL: {np.mean(losses):.3f}/{current_ppl:.3f}")

        return losses
    
    def generate(self, examples, retrieved_context, max_generation_length):
        """
        Generates.

        Args:
            examples: A collection of examples.
            retrieved_context: Retrieved context.
            max_generation_length: Maximum length of generation.

        Returns:
            A list of generated answer.
        """
        generated_answer = []
        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_context,generation=True)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)
        if hasattr(self.model, "module"):
            self.model.module.max_generation_length = max_generation_length
        else:
            self.model.max_generation_length = max_generation_length

        pbar = tqdm(dataloader, disable=not self.args.enable_tqdm, desc="Generating")
        with torch.no_grad():
            for batch in pbar:
                generated_answer.append(self.model(batch.cuda()))
        generated_answer = torch.cat(generated_answer,0)
        return  [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_answer]


