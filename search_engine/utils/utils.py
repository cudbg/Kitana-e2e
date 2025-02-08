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

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_and_save_datasets():
    buyer_dir = 'data/test_data_small/buyer'
    seller_dir = 'data/test_data_small/seller'
    
    # Ensure the existence of the target dir
    ensure_dir(buyer_dir)
    ensure_dir(seller_dir)
    
    
    buyer_rows = 500
    buyer_features = 100
    buyer_join_key_domains = {'m': 1000,
                              'n': 1000}
    buyer_df, buyer_features_cols = create_dataframe(
        rows=buyer_rows, 
        features=buyer_features, 
        join_key_domains=buyer_join_key_domains, 
        prefix='b'
    )
    buyer_file_path = os.path.join(buyer_dir, 'buyer.csv')
    buyer_df.to_csv(buyer_file_path, index=False)
    print(f"Buyer dataset saved to {buyer_file_path}")
    
    seller_rows = 300
    seller_features = 50
    seller_join_key_domains = {'m': 1000,
                               'n': 1000}
    for i in range(1, 4):
        seller_df, seller_features_cols = create_dataframe(
            rows=seller_rows, 
            features=seller_features, 
            join_key_domains=seller_join_key_domains, 
            prefix=f's{i}_'
        )
        seller_file_path = os.path.join(seller_dir, f'seller_{i}.csv')
        seller_df.to_csv(seller_file_path, index=False)
        print(f"Seller dataset {i} saved to {seller_file_path}")

if __name__ == '__main__':
    generate_and_save_datasets()
