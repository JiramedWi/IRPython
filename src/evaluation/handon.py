import numpy as np

from collections import Counter
from functools import reduce
import operator

# Given an automated system used to rank reported bugs, where the most critical bugs should be addressed first.
# Suppose there are 5 bugs, all deemed critical.
# Example of ranking system Critical = 3, Major = 2, Minor = 1, No = 0 and k = 5
# Find NDCG @ k by receive input of ranking system and k


system_a = [1, 3, 3, 3, 3, 1, 3, 1, 1, 1]
system_b = [3, 3, 3, 1, 1, 1, 1, 3, 3, 1]

system_c = [3, 3, 2, 2, 2, 2, 1, 0, 0, 0]
k = 5

import numpy as np


def dcg_at_k(relevances, k):
    """Compute DCG at K."""
    relevances = np.array(relevances[:k])
    discounts = np.log2(np.arange(2, k + 2))  # log2(1) is undefined, so we start from log2(2)
    return np.sum(relevances / discounts)


def ndcg_at_k(system_ranking, k):
    """Compute NDCG at K."""
    dcg = dcg_at_k(system_ranking, k)
    ideal_ranking = sorted(system_ranking, reverse=True)  # Best possible ranking
    idcg = dcg_at_k(ideal_ranking, k)
    return dcg / idcg if idcg > 0 else 0


ndcg_a_5 = ndcg_at_k(system_a, 5)
ndcg_b_5 = ndcg_at_k(system_b, 5)
ndcg_c_5 = ndcg_at_k(system_c, 5)


def sequence_probability(possible_values, target_sequence):
    k = len(target_sequence)
    num_choices = len(possible_values)
    return (1 / num_choices) ** k * 100


# Example usage
possible_values = ['C', 'B', 'A', 'S', 'SS']
set_a = ['C', 'C', 'C', 'C']
set_b = ['C', 'C', 'C', 'C', 'C']
prob = sequence_probability(possible_values, set_a)
print(f"Probability of {set_a}: {prob:.2f}%")
prob = sequence_probability(possible_values, set_b)
print(f"Probability of {set_b}: {prob:.2f}%")
