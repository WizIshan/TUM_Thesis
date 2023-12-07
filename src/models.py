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

from ft_datasets import FTDataset


class FineTuner():

    def __init__(self,model_name = None, from_local = False, local_model_path = None, model_save_dir = None):

        '''
        Initializes local variables for the class FineTuner 
        '''
        
        self.model_checkpoint = model_name
        self.model = None
        self.tokenizer = None

        if(self.model_checkpoint == None):
            print("No Model name specified!")
        else:
            self.__initialize_model__(from_local,local_model_path)

        self.model_save_dir = model_save_dir

    def __initialize_model__(self,from_local,local_model_path):

        '''
        This function intializes the model by either downloading the weights from huggingface
        or loading pre-trained weights from the local models directory. 
        '''
        print("Initializing Model")
        if(from_local):
            self.model = AutoModelForMaskedLM.from_pretrained(local_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        print("Model Initialized!")
    
    def getModel(self):

        '''
        Returns the loaded model
        '''
        return self.model

    def getTokenizer(self):

        '''
        Returns the loaded Tokenizer
        '''
        return self.tokenizer

    def finetune_model(self, dataset_name = None):

        '''
        This function fine tunes the chosen model on the dataset specified by the user.
        The new model is saved in the local model directory.
        '''
        if(dataset_name == None):
            print("Did not specify the name of dataset!")
            return
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)

        data = FTDataset(dataset_name,self.model,self.tokenizer)
        downsampled_dataset = data.generate_dataset()
        
        ## Create the dataset and format it for training

        batch_size = 64
        # Show the training loss with every epoch
        logging_steps = len(downsampled_dataset["train"]) // batch_size

        training_args = TrainingArguments(
            output_dir=self.model_save_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            num_train_epochs = 10,
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            push_to_hub=False,
            fp16=True,
            logging_steps=logging_steps,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=downsampled_dataset["train"],
            eval_dataset=downsampled_dataset["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            )
        
        trainer.train()

        return
        