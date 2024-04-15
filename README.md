# TUM_Thesis
## Studying Bias with Contextual Embeddings- The Influence of Pretraining and Finetuning

To install libraries use

```
pip install -r requirments.txt
```

Set the current working directory in paths.py 

For fine-tuning models use finetune.ipynb

To generate results run main.ipynb.

plots.ipynb is used to generate visualizations from the generated results.

The data directory structure should be as follows : 

```
data/
 - ft_ds
 - models
 - metrics_ds
    - ceat
    - crows-pairs
    - stereoset
 - results
    - ceat

```

1. ft_ds : Data related to the datasets used for fine-tuning like raw text files, etc.
2. models : Saved models with checkpoints
3. metrics_ds : Data related to the bias metrics. Can be datasets used for generating results or for creating the datasets used for evaluation.
4. results : All the results from the different setups and metrics are stored here.



