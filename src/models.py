import torch
import torch.nn as nn

import transformers

from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling, default_data_collator, TrainingArguments, Trainer

from datasets import load_dataset
import evaluate

import random
import os
import pandas as pd
import numpy as np
import collections
from datetime import datetime
from ft_datasets import FTDataset
from paths import *


class FineTuner():

    '''
    This class is used to load pre-trained models and fine tune them on
    the user provided dataset.
    '''

    def __init__(self,model_name = None, from_local = False, local_model_path = None, random_init = False, **kwargs):

        '''
        Initializes local variables for the class FineTuner 
        '''
        
        self.model_checkpoint = model_name
        self.model = None
        self.tokenizer = None
        self.model_tag = model_name
        # self.model_save_dir = model_save_dir

        if(self.model_checkpoint == None):
            print("No Model name specified!")
        
        else:

            if(random_init):
                self.__random_initialize_model__()
                self.model_dir = 'random init'
            else:
                
                if(from_local):
                    self.model_dir = local_model_path
                    # self.model_tag = self.model_tag + '_FT_' + kwargs['ft_ds']
                    if('random' in self.model_dir):
                        self.model_tag = self.model_tag + '_random_init'
                    if('FT' in self.model_dir):
                        self.model_tag = self.model_tag + '_FT_' + kwargs['ft_ds']


                else:
                    self.model_dir = None
                
                self.__initialize_model__(from_local,local_model_path)


    def __random_initialize_model__(self):

        '''
        This function is used to randomly initialize the weights of the given model.
        '''

        self.model = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(self.model_checkpoint))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        self.model_tag = self.model_tag + '_random_init'

        print("Model Initialized with random weights!")


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
            # self.model_dir = 'None'
        print("Model Initialized!")
    
    def save_model(self):

        '''
        This function is used to save the model parameters locally.
        '''
        model_path = os.path.join(MODELS_PATH,(self.model_tag+ "_"+(datetime.now()).strftime("%Y-%m-%d-%H-%M-%S")))
        print("Model saved in : ", model_path)
        self.model_dir = model_path

        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)



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

    def finetune_model(self, dataset_name = None, model_save_dir = None, epochs = 20, train_size = 10_000):

        '''
        This function fine tunes the chosen model on the dataset specified by the user.
        The new model is saved in the local model directory.
        '''
        if(dataset_name == None):
            print("Did not specify the name of dataset!")
            return
        
        ## Modify model tag
        self.model_tag = self.model_tag + '_FT_' + dataset_name
        # if(model_save_dir):
        #     self.model_save_dir = model_save_dir
        #     self.model_dir = model_save_dir
        
        model_save_dir = os.path.join(MODELS_PATH,(self.model_tag+ "_"+(datetime.now()).strftime("%Y-%m-%d-%H-%M-%S")))
        self.model_dir = model_save_dir
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)

        data = FTDataset(dataset_name,self.model,self.tokenizer,train_size)
        downsampled_dataset = data.generate_dataset()
        
        ## Create the dataset and format it for training

        batch_size = 64
        # Show the training loss with every epoch
        logging_steps = len(downsampled_dataset["train"]) // batch_size

        training_args = TrainingArguments(
            output_dir = model_save_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            num_train_epochs = epochs,
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            push_to_hub=False,
            fp16=True,
            logging_steps=logging_steps,
            save_strategy = "steps",
            # save_only_model = True
            save_steps = logging_steps * 5 ## Save after every 5 epochs,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            # train_dataset=downsampled_dataset["train"].with_format("torch"),
            # eval_dataset=downsampled_dataset["test"].with_format("torch"),
            train_dataset=downsampled_dataset["train"],
            eval_dataset=downsampled_dataset["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            )
        
        trainer.train()

        return
        