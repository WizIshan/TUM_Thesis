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
from metrics.ceat.ceat import *

from datetime import datetime
import os


class Metric():

    '''
    This class is used for generating the metrics for a given MLM.
    '''

    def __init__(self, model_name = None , model = None, tokenizer = None, model_tag = None, model_dir = None) -> None:
        
        '''
        Initializes variables for the Metric class.
        '''
    
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model_tag = model_tag
        self.model_dir = model_dir


    def __crows_pair_metric__(self, input_file = None, output_file = None, output_dir = None):

        '''
        Generates crows-pairs metrics for a given MLM.
        '''

        return get_results(input_file, output_file, self.model, self.tokenizer)


    
    def __stereoset_metric__(self, input_file = None, output_file = None, output_dir = None):

        '''
        Generates bias metrics on StereoSet data for a given MLM.
        '''

        return getStereoSet(pretrained_class =  self.model_name, tokenizer = self.tokenizer, 
                            intrasentence_model =  self.model, 
                            input_file = input_file, 
                            output_dir = output_dir,
                            output_file = output_file )
    
    def __ceat_metric__(self, input_dir = None, output_dir = None, generate_new = False):

        '''
        Generates CEAT score for a given MLM
        '''

        return get_ceat(input_dir, output_dir, self.model,self.tokenizer, exp_name = self.model_tag, generate_new = generate_new)
        

    def save_metric(self, metric = None, score = None, file = None):

        '''
        Saves the bias metric in a common file with relevant details.
        '''
        ## Adding metadata for runs
        score = pd.DataFrame(score)
        score['ts'] = (datetime.now()).strftime("%Y-%m-%d")
        score['model_name'] = self.model_tag
        score['model_dir'] = self.model_dir
        score['metric'] = metric

        ## Creating consolidated data
        cons_df = {'ts' : [(datetime.now()).strftime("%Y-%m-%d")], 'model_tag' : [self.model_tag], 'model_dir' : [self.model_dir], 'metric' : [metric], 'score' : []}
        if(metric == 'crows-pairs'):
            cons_df['score'].append(score['metric_score'].values[0])
        elif(metric == 'stereoset'):
            cons_df['score'].append(score.loc[score['category']=='overall']['ICAT Score'].values[0])
        elif(metric == 'ceat'):
            cons_df['score'].append(score.loc[score['group']==0]['PES'].values[0])
        
        # print(pd.DataFrame(cons_df))
        cons_df = pd.DataFrame(cons_df)

        if(os.path.isfile(file)):        
            original_metrics = pd.read_excel(file, sheet_name=None, index_col = None)
            writer = pd.ExcelWriter(file, engine = 'xlsxwriter')
            if(not metric in original_metrics.keys()):
                score.to_excel(writer, sheet_name = metric, index=False)
            for sheet in original_metrics.keys():
                if(sheet == metric):
                    new_metrics = pd.concat([original_metrics[sheet],score])
                    new_metrics.to_excel(writer, sheet_name = sheet, index=False)
                elif(sheet == 'consolidated'):
                    new_metrics = pd.concat([original_metrics['consolidated'],cons_df])
                    new_metrics.to_excel(writer, sheet_name = 'consolidated', index=False)
                else:
                    original_metrics[sheet].to_excel(writer, sheet_name = sheet, index=False)
            
            writer.close()
        else:
            # print(score)
            writer = pd.ExcelWriter(file, engine = 'xlsxwriter')
            score.to_excel(writer, sheet_name = metric, index=False)
            cons_df.to_excel(writer, sheet_name = 'consolidated', index = False)
            writer.close()

        print(f"Metrics Saved in ", file)

        

    def get_metric(self, metric = None, input_file = None, output_file = None, output_dir = None, **kwargs):

        '''
        Returns metrics specified in the input. 
        '''

        if(metric == 'all'):

            ## Call all functions one by one
            score = self.__crows_pair_metric__(input_file, output_file)
            self.save_metric(metric = 'crows-pairs', score = score, file = '/home/bhatt/ishan/TUM_Thesis/data/results/master_results.xlsx')
            score = self.__stereoset_metric__(input_file, output_file, output_dir)
            self.save_metric(metric = 'stereoset', score = score, file = '/home/bhatt/ishan/TUM_Thesis/data/results/master_results.xlsx')

            score = self.__ceat_metric__(kwargs['input_dir'], output_dir, kwargs['generate_new'])
            self.save_metric(metric = 'ceat', score = score, file = '/home/bhatt/ishan/TUM_Thesis/data/results/master_results.xlsx')   



        elif(metric == 'crows-pairs'):

            ## Call crows-pairs function
            score = self.__crows_pair_metric__(input_file, output_file)
            self.save_metric(metric = 'crows-pairs', score = score, file = '/home/bhatt/ishan/TUM_Thesis/data/results/master_results.xlsx')

        elif(metric == 'stereoset'):

            ## Call stereoset function
            score = self.__stereoset_metric__(input_file, output_file, output_dir)
            # print(score)
            self.save_metric(metric = 'stereoset', score = score, file = '/home/bhatt/ishan/TUM_Thesis/data/results/master_results.xlsx')

        elif(metric == 'ceat'):

            ##Call CEAT function
            score = self.__ceat_metric__(kwargs['input_dir'], output_dir, kwargs['generate_new'])
            self.save_metric(metric = 'ceat', score = score, file = '/home/bhatt/ishan/TUM_Thesis/data/results/master_results.xlsx')
        
        else:

            print("Metric not available!")

    
    

    