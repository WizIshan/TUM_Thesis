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


class FineTuner():

    def __init__(self,model_name = None, from_local = False, local_model_path = None, model_save_dir = None):
        
        self.model_checkpoint = model_name
        self.model = None
        self.tokenizer = None

        if(model_name == None):
            print("No Model name specified!")
        else:
            self.model = self.__initialize_model__(from_local,local_model_path)


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

    def finetune_model(self):

        '''
        This function fine tunes the chosen model on the dataset specified by the user.
        The new model is saved in the local model directory.
        '''



        return
        