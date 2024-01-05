'''Utility functions for experiments.
'''
import os
import random

import numpy as np

import torch

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)


def set_seed(seed=42):
    '''Set seed across all platforms to same value.
    
    Arguments:
        seed (int): Random seed.
    '''
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_metrics(label_pairs):
    '''Compute evaluation metrics.
    
    Arguments:
        label_pairs (dict): Tracks ground-truth and prediction pairs for
            both backward and forward slices, for all instances.
    
    Returns:
        metrics (dict): Evaluation metrics.
    '''
    metrics = {}
    for slice_type in ['back', 'forward']:
        true = label_pairs[slice_type]['true']
        flattened_true = [item for sublist in true for item in sublist]

        pred = label_pairs[slice_type]['preds']
        flattened_pred = [item for sublist in pred for item in sublist]
        
        em_accuracy = sum([1 for _id, item_true in enumerate(true) \
                           if item_true == pred[_id]]) / len(true)
        metrics[slice_type.upper()] = {
            'EM-Accuracy': em_accuracy,
            'Accuracy': accuracy_score(flattened_true, flattened_pred),
            'Precision': precision_score(flattened_true, flattened_pred),
            'Recall': recall_score(flattened_true, flattened_pred),
            'F1-Score': f1_score(flattened_true, flattened_pred),
        }

    total_true = label_pairs['back']['true'] + label_pairs['forward']['true']
    total_flattened_true = [item for sublist in total_true for item in sublist]

    total_pred = label_pairs['back']['preds'] + label_pairs['forward']['preds']
    total_flattened_pred = [item for sublist in total_pred for item in sublist]

    total_em_accuracy = sum([1 for _id, item_true in enumerate(total_true) \
                             if item_true == total_pred[_id]]) / len(total_true)

    metrics['OVERALL'] = {
        'EM-Accuracy': total_em_accuracy,
        'Accuracy': accuracy_score(total_flattened_true, total_flattened_pred),
        'Precision': precision_score(total_flattened_true, total_flattened_pred),
        'Recall': recall_score(total_flattened_true, total_flattened_pred),
        'F1-Score': f1_score(total_flattened_true, total_flattened_pred),
    }
    return metrics


def display_stats(stats, message=None):
    '''Print evaluation metrics.
    
    Arguments:
        stats (dict): Evaluation statistics.
        message (str): Evaluation setting heading.
    '''
    if message:
        print(message)
        print("-" * len(message))

    print("  Test loss: {0:.4f}".format(stats['Epoch evaluation loss']))
    for key in ['Back', 'Forward', 'Overall']:
        print(f"  {key} Slice Accuracy: {stats[key.upper()]['Accuracy']:.4f}")
        print(f"  {key} Slice Precision: {stats[key.upper()]['Accuracy']:.4f}")
        print(f"  {key} Slice Recall: {stats[key.upper()]['Accuracy']:.4f}")
        print(f"  {key} F1-Score: {stats[key.upper()]['F1-Score']:.4f}")
        print(f"  {key} Slice EM-Accuracy: {stats[key.upper()]['EM-Accuracy']:.4f}")
        print()
