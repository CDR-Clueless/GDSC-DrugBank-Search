#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 3 Nov 2026

@author: jds40
"""

import random

def split_list(l: list, parts: int, shuffle: bool = False) -> list:
    """Split a larger list into a given number of component lists (used here for more efficient multiprocessing batches)

    Args:
        l (list): A list
        parts (int): Number of component lists to break the larger list into
        shuffle (bool): Whether the components of the lists should be shuffled (Defaults to False; not shuffling is also more computationally efficient)

    Returns:
        list: List of output lists all of equal size
    """
    # Number of parts to break this into; if we want to split it into more parts than there are, we'll need to add blank lists
    n = min(parts, max(len(l),1))
    if(shuffle == False):
        n = min(parts, max(len(l),1))
        k, m = divmod(len(l), n)
        output = [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
        # Add extra empty lists if needed
        while(n<parts):
            output.append([])
            n += 1
    else:
        # Shuffle which parts go where
        output = [[] for _ in range(n)]
        choices = [i for i in range(n)]
        for item in l:
            # Decide which output sub-list to put this item in
            choice = random.choice(choices)
            output[choice].append(item)
            # If the sub-list is now longer than the length of the original list divided by number of lists, remove that sub-list as an option for future choices
            if(len(output[choice]) > len(l) / n):
                choices.remove(choice)
    return output
