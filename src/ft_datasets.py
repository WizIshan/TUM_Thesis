import torch
import torch.nn as nn

import transformers

from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator, TrainingArguments, Trainer

from datasets import load_dataset
import evaluate

import random

import pandas as pd
import numpy as np
import collections


class FTDataset():

    '''
    This class is used to download datasets from HuggingFace and format them so that they
    can be used for Fine Tuning Masked Language Models.
    '''

    def __init__(self, dataset_name, model, tokenizer, train_size = 10_000):

        '''
        This function initiliazes the global variables for the class FTDataset
        '''
        
        self.dataset_name = dataset_name
        self.model = model
        self.tokenizer = tokenizer
        self.train_size = train_size
    

    def tokenize_function(self,examples):

        '''
        Tokenizes the words in the dataset
        '''

        result = self.tokenizer(examples["text"])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    
    def group_texts(self,examples, chunk_size = 128):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result
    
    def whole_word_masking_data_collator(self, features, wwm_probability = 0.2):

        '''
        Data collator for whole word masking of input sequence.
        '''

        for feature in features:
            word_ids = feature.pop("word_ids")

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            mask = np.random.binomial(1, wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = self.tokenizer.mask_token_id
            feature["labels"] = new_labels

        return default_data_collator(features)
    
    def generate_dataset(self):

        '''
        Generates the formatted dataset to use for fine-tuning.
        '''
        if(self.dataset_name == 'imdb'):

            dataset = load_dataset(self.dataset_name)
            # Use batched=True to activate fast multithreading!
            tokenized_datasets = dataset.map(
                self.tokenize_function, batched=True, remove_columns=["text", "label"]
            )
        
        elif(self.dataset_name == '4chan'):

            dataset = load_dataset("json", data_files="/home/bhatt/ishan/TUM_Thesis/data/ft_ds/4chan/pol-dataset.json",streaming=True)
            # Use batched=True to activate fast multithreading!
            tokenized_datasets = dataset.map(
                self.tokenize_function, batched=True, remove_columns=['author', 'id', 'token_length', 'text']
            )
        

        lm_datasets = tokenized_datasets.map(self.group_texts, batched=True)
 

        test_size = int(0.1 * self.train_size)

        downsampled_dataset = lm_datasets["train"].train_test_split(
            train_size=self.train_size, test_size=test_size, seed=42
        )

        return downsampled_dataset