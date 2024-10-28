import torch
import os
import numpy as np
import pandas as pd

def cleanup(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor):
            del arg
    torch.cuda.empty_cache()

def create_dataframe(rows=1000, features=10000, join_key_domains=None, prefix='f'):
    if join_key_domains is None:
        join_key_domains = {'join_key': 1000}

    data = np.random.randint(low=0, high=100, size=(rows, features))
    feature_cols = [f'{prefix}{i+1}' for i in range(features)]

    df = pd.DataFrame(data, columns=feature_cols)

    for key, domain in join_key_domains.items():
        join_keys = np.random.choice(
            domain, size=rows, replace=True)
        df.insert(0, key, join_keys)

    return df, feature_cols