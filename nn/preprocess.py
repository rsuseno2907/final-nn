# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    positives = [s for s, l in zip(seqs, labels) if l]
    negatives = [s for s, l in zip(seqs, labels) if not l]
    n_pos = len(positives)
    n_neg = len(negatives)
    
    # Balance to the size of the majority class
    target_size = max(n_pos, n_neg)
    
    # Oversample with replacement
    sampled_pos = random.choices(positives, k=target_size)
    sampled_neg = random.choices(negatives, k=target_size)
    
    # Combine
    sampled_seqs = sampled_pos + sampled_neg
    sampled_labels = [True] * target_size + [False] * target_size
    # print('meow')
    # print(len(sampled_seqs))
    
    # Shuffle so classes aren't grouped
    combined = list(zip(sampled_seqs, sampled_labels))
    random.shuffle(combined)
    
    sampled_seqs, sampled_labels = zip(*combined)
    
    return list(sampled_seqs), list(sampled_labels)

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    mapping = {"A":[1,0,0,0],"T":[0,1,0,0],"C":[0,0,1,0],"G":[0,0,0,1],"N":[0,0,0,0]}
    encodings = []

    # loop through each nucleotide
    for seq in seq_arr:
        one_hot = []
        for base in seq.upper():
            one_hot.extend(mapping[base])
        encodings.extend(one_hot)
        
    return encodings