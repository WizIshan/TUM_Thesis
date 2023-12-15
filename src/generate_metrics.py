import torch
import torch.nn as nn

import transformers

import evaluate

import random

import pandas as pd
import numpy as np
import collections

from metrics.crows_pairs import *
from metrics.stereoset.eval_discriminative_models import *


class Metric():

    '''
    This class is used for generating the metrics for a given MLM.
    '''

    def __init__(self, model_name = None , model = None, tokenizer = None) -> None:
        
        '''
        Initializes variables for the Metric class.
        '''
    
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name


    def __crows_pair_metric__(self, input_file = None, output_file = None, output_dir = None):

        get_results(input_file, output_file, self.model, self.tokenizer)
    
    def __stereoset_metric__(self, input_file = None, output_file = None, output_dir = None):

        getStereoSet(pretrained_class =  self.model_name, tokenizer = self.tokenizer, 
             intrasentence_model =  self.model, 
             input_file = input_file, 
             output_dir = output_dir,
              output_file = output_file )
        
    def get_metric(self,metric = None, input_file = None, output_file = None, output_dir = None):

        if(metric == 'all'):

            ## Call all functions one by one
            self.__crows_pair_metric__(input_file, output_file)
            self.__stereoset_metric__(input_file, output_file, output_dir)
        
        elif(metric == 'crows-pairs'):

            ## Call crows-pairs function
            self.__crows_pair_metric__(input_file, output_file)

        elif(metric == 'stereoset'):

            ## Call stereoset function
            self.__stereoset_metric__(input_file, output_file, output_dir)

    