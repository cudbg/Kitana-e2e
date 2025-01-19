import pandas as pd
import torch
import logging
from ..market.data_market import DataMarket
from ..utils.logging_utils import log_execution, handle_exceptions
from ..config.config import get_config

def cleanup(*args):
    """Helper function to clean up GPU memory"""
    for arg in args:
        if isinstance(arg, torch.Tensor):
            del arg
    torch.cuda.empty_cache()

class SearchEngine():
    """
    A search engine class that helps find the best augmentation features 
    from seller datasets for a buyer's target prediction task.
    """
    @handle_exceptions
    @log_execution(logging.INFO)
    def __init__(self, data_market: DataMarket):
        config = get_config()
        self.fit_by_residual = config.search.fit_by_residual
        self.device = config.search.device
        self.batch_size = config.search.batch_size
        self.augplan = []
        self.augplan_acc = []
        self.aug_seller_feature_ind = {}
        self.buyer_target = data_market.buyer_target_feature
        self.buyer_features = data_market.buyer_dataset.columns
        self.buyer_dataset = None  
        self.buyer_sketches = {}  
        self.seller_sketches = {}  
        self.data_market = data_market
        self.seller_aggregated = {}
        self.unusable_features = {}

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def search_one_iteration(self):
        """Search for the best seller feature in one iteration."""
        best_r_squared = 0
        best_r_squared_ind = -1
        best_batch_id = -1
        best_join_key = None

        self.buyer_sketches = self.data_market.buyer_sketches
        logging.debug(f"Starting search_one_iteration with {self.buyer_sketches.keys()} buyer sketches")
        for join_key in self.buyer_sketches.keys():
            buyer_id = self.buyer_sketches[join_key]["id"]
            buyer_join_key_domain = self.buyer_sketches[join_key]["join_key_domain"]
            buyer_sketch = self.buyer_sketches[join_key]["buyer_sketch"]

            buyer_1, buyer_y, buyer_y_y, buyer_x_y = buyer_sketch.get_sketches()
            search_sketch_base = self.data_market.get_seller_sketch_base_by_keys(join_key)

            for batch_id in range(search_sketch_base.get_sketch_loader().get_num_batches()):
                seller_1, seller_x, seller_x_x, _ = search_sketch_base.get_sketch_loader().get_sketches(batch_id)
                
                if not self.fit_by_residual:
                    d = buyer_y.shape[1]
                    ordered_columns = buyer_sketch.get_base().dfid_feature_mapping[buyer_id]
                    y_ind = ordered_columns.index(buyer_sketch.get_target_feature()["name"])
                    
                    XTX = torch.zeros(seller_x.shape[1], d+1, d+1).to(self.data_market.device)
                    XTY = torch.zeros(seller_x.shape[1], d+1, 1).to(self.data_market.device)
                    c = torch.sum(buyer_1 * seller_1, dim=0)
                    x = torch.sum(seller_x * buyer_1, dim=0)
                    x_x = torch.sum(seller_x_x * buyer_1, dim=0)
                    x_x[x_x == 0] = 1
                    y = torch.sum(buyer_y[:, y_ind:y_ind+1] * seller_1, dim=0) 
                    y_y = torch.sum(buyer_y_y[:, y_ind:y_ind+1] * seller_1, dim=0)
                    TSS = y_y - y * y / c

                    XTX[:, 0, 0] = c
                    XTX[:, 0, 1] = XTX[:, 1, 0] = x
                    XTX[:, 1, 1] = x_x

                    for i in range(d):
                        cur_buyer_y = buyer_y[:, i:i+1]
                        cur_buyer_y_y = buyer_y_y[:, i:i+1]
                        cur_x_y = torch.sum(seller_x * cur_buyer_y, dim=0)
                        cur_y_y = torch.sum(cur_buyer_y_y * seller_1, dim=0)
                        cur_y = torch.sum(cur_buyer_y * seller_1, dim=0)
                        cur_y_y[cur_y_y == 0] = 1

                        if i == y_ind:
                            XTY[:, 0, 0] = cur_y
                            XTY[:, 1, 0] = cur_x_y
                        elif i < y_ind:
                            XTX[:, i+2, i+2] = cur_y_y
                            XTX[:, 1, i+2] = XTX[:, i+2, 1] = cur_x_y
                            XTX[:, 0, i+2] = XTX[:, i+2, 0] = cur_y
                        else:
                            XTX[:, i+1, i+1] = cur_y_y
                            XTX[:, 1, i+1] = XTX[:, i+1, 1] = cur_x_y
                            XTX[:, 0, i+1] = XTX[:, i+1, 0] = cur_y
                        
                        for j in range(i+1, d):
                            x_y_ind = int((2*d-i-1)*i/2+j-i)-1
                            x_y_ = torch.sum(buyer_x_y[:, x_y_ind:x_y_ind+1] * seller_1, dim=0)
                            if i == y_ind:
                                XTY[:, j+1, 0] = x_y_
                            elif j == y_ind:
                                XTY[:, i+2, 0] = x_y_
                            elif i > y_ind:
                                XTX[:, i+1, j+1] = XTX[:, j+1, i+1] = x_y_
                            elif i < y_ind and j > y_ind:
                                XTX[:, i+2, j+1] = XTX[:, j+1, i+2] = x_y_
                            else:
                                XTX[:, i+2, j+2] = XTX[:, j+2, i+2] = x_y_

                    inverses = torch.empty_like(XTX)

                    for i in range(len(XTX)):
                        try:
                            inverses[i] = torch.linalg.inv(XTX[i])
                        except RuntimeError as e:
                            logging.warning(f"Singular matrix at batch {batch_id} and feature index {i}")
                            self.unusable_features[batch_id] = self.unusable_features.get(batch_id, [])
                            self.unusable_features[batch_id].append(i)
                            inverses[i] = torch.zeros_like(XTX[i])

                    res = torch.bmm(inverses, XTY).to(self.data_market.device)
                    RSS = y_y
                    for i in range(d+1):
                        for j in range(d+1):
                            RSS += res[:, i, 0]*res[:, j, 0]*XTX[:, i, j]
                        RSS -= 2*res[:, i, 0]*XTY[:, i, 0]
                    r_squared = 1 - RSS / TSS

                else:
                    x_x = torch.sum(seller_x_x * buyer_1, dim=0)
                    x = torch.sum(seller_x * buyer_1, dim=0)
                    c = torch.sum(buyer_1 * seller_1, dim=0)
                    x_y = torch.sum(seller_x * buyer_y, dim=0)
                    y_y = torch.sum(buyer_y_y * seller_1, dim=0)
                    y = torch.sum(buyer_y * seller_1, dim=0)

                    x_mean = x / c
                    y_mean = y / c

                    S_xx = x_x - 2 * x_mean * x + c * x_mean ** 2
                    S_xy = x_y - x_mean * y - x * y_mean + c * x_mean * y_mean

                    slope = S_xy / S_xx
                    intercept = y_mean - slope * x_mean

                    TSS = y_y - 2 * y_mean * y + c * y_mean ** 2
                    RSS = y_y + c * intercept ** 2 + slope ** 2 * x_x - 2 * \
                        (slope * x_y + intercept * y - slope * intercept * x)

                    r_squared = 1 - (RSS / TSS)

                r_squared = torch.where(torch.isnan(r_squared), torch.tensor(float('-inf')), r_squared)
                r_squared = torch.where(r_squared >= 1, torch.tensor(float('-inf')), r_squared)

                if batch_id in self.unusable_features:
                    for singular_ind in self.unusable_features[batch_id]:
                        r_squared[singular_ind] = float('-inf')

                if join_key in self.aug_seller_feature_ind and batch_id in self.aug_seller_feature_ind[join_key]:
                    exclude_indices = self.aug_seller_feature_ind[join_key][batch_id]
                    original_values = r_squared[exclude_indices].clone()
                    r_squared[exclude_indices] = float('-inf')
                    max_r2_index = torch.argmax(r_squared)

                    if r_squared[max_r2_index].item() < -1:
                        r_squared[exclude_indices] = original_values
                        continue

                    r_squared[exclude_indices] = original_values
                else:
                    max_r2_index = torch.argmax(r_squared)

                if r_squared[max_r2_index].item() > best_r_squared:
                    best_r_squared = r_squared[max_r2_index].item()
                    best_r_squared_ind = max_r2_index
                    best_batch_id = batch_id
                    best_join_key = join_key

                if not self.fit_by_residual:
                    cleanup(x_x, x, c, y, y_y, inverses, res, TSS,
                            RSS, r_squared, seller_1, seller_x, seller_x_x)
                else:
                    cleanup(x_x, x, c, y, y_y, x_y, x_mean, y_mean, S_xx, S_xy, TSS,
                            RSS, r_squared, slope, intercept, seller_1, seller_x, seller_x_x)
                    
        if best_r_squared_ind == -1:
            return None, None, None
        else: 
            logging.info(f"Maximum R² value: {best_r_squared}")
            return best_join_key, best_r_squared_ind.item(), best_batch_id

    @handle_exceptions
    @log_execution(logging.INFO)
    def start(self, iter=2):
        """Start the search engine and iterate through multiple rounds."""
        for i in range(iter):
            join_key, ind, batch_id = self.search_one_iteration()
            if not join_key:
                logging.info("No more good features")
                break

            if join_key not in self.aug_seller_feature_ind:
                self.aug_seller_feature_ind[join_key] = {batch_id: torch.tensor([ind])}
            elif batch_id not in self.aug_seller_feature_ind[join_key]:
                self.aug_seller_feature_ind[join_key][batch_id] = torch.tensor([ind])
            else:
                self.aug_seller_feature_ind[join_key][batch_id] = torch.cat(
                    (self.aug_seller_feature_ind[join_key][batch_id], torch.tensor([ind])))

            seller_id, best_feature = self.data_market.get_seller_sketch_base_by_keys(join_key).get_df_by_feature_index(
                batch_id, ind)
            logging.info(f"The best feature in iter {i} is: {best_feature} with join key {join_key}")
            
            self.augplan.append((seller_id, 
                                i+1, 
                                self.data_market.seller_id_to_df_and_name[seller_id]["name"],
                                best_feature))
            self._update_residual(join_key, seller_id, best_feature)
        logging.debug(f"Augmented plan: {self.augplan}")
        return self.augplan, self.data_market.augplan_acc, self.data_market.buyer_dataset if not self.fit_by_residual else self.data_market.buyer_dataset_for_residual

    @handle_exceptions
    @log_execution(logging.DEBUG)    
    def _update_residual(self, join_key, seller_id, best_feature):
        """Update the residual after finding a new feature."""
        logging.info(f"Starting _update_residual with join_key={join_key}, seller_id={seller_id}, best_feature={best_feature}")
        # Buyer data info
        buyer = self.data_market.buyer_id_to_df_and_name[0]["dataframe"]
        if self.fit_by_residual:
            buyer = self.data_market.buyer_dataset_for_residual
        logging.debug(f"Buyer DataFrame shape: {buyer.shape}, columns: {buyer.columns.tolist()}")
        
        # Seller data info
        seller_sketch = self.data_market.get_seller_sketch_by_keys(join_key=join_key, seller_id=seller_id)
        seller_df = seller_sketch.get_df()
        logging.debug(f"Original Seller DataFrame shape: {seller_df.shape}, columns: {seller_df.columns.tolist()}")
        
        # Select columns and check
        seller_df = seller_df[[join_key, best_feature]]
        logging.debug(f"Selected Seller DataFrame shape: {seller_df.shape}")
        logging.debug(f"Seller data sample:\n{seller_df.head()}")
        
        # Aggregation info
        aggregation_functions = {col: 'mean' for col in seller_df.columns if col != join_key}
        logging.debug(f"Aggregation functions: {aggregation_functions}")
        
        seller_df_agg = seller_df.groupby(join_key).agg(aggregation_functions).reset_index()
        logging.debug(f"Aggregated Seller DataFrame shape: {seller_df_agg.shape}")
        logging.debug(f"Aggregated data sample:\n{seller_df_agg.head()}")
        
        # Merge info
        joined_df = pd.merge(buyer, seller_df_agg, how='left', on=join_key, 
                            suffixes=('_KitanaSearchLeft', '_KitanaSearchRight'))
        logging.debug(f"Joined DataFrame shape: {joined_df.shape}, columns: {joined_df.columns.tolist()}")
        
        # Clean up columns
        joined_df = joined_df[[col for col in joined_df.columns if '_KitanaSearchRight' not in col]]
        joined_df.columns = [col.replace('_KitanaSearchLeft', '') for col in joined_df.columns]
        logging.debug(f"Cleaned DataFrame shape: {joined_df.shape}, columns: {joined_df.columns.tolist()}")
        
        # Handle missing values
        for col in seller_df_agg.columns:
            if col != join_key:
                null_count = joined_df[col].isnull().sum()
                if null_count > 0:
                    mean_val = joined_df[col].mean()
                    logging.debug(f"Filling {null_count} null values in column {col} with mean value {mean_val}")
                    joined_df[col].fillna(mean_val, inplace=True)
        
        # Final update info
        updated_buyer = joined_df[list(set([join_key] + list(buyer.columns) + [best_feature] + [self.buyer_target]))]
        logging.info(f"Final updated buyer DataFrame shape: {updated_buyer.shape}, columns: {updated_buyer.columns.tolist()}")
    
        buy_keys = self.data_market.buyer_join_keys
        join_key_domain = self.data_market.buyer_sketches[join_key]["join_key_domain"]

        self.data_market.set_buyer_id(0)
        self.data_market.reset_buyer_sketches()
        self.data_market.reset_buyer_id_to_df_and_name()
        self.data_market.register_buyer(updated_buyer, buy_keys,
                                      join_key_domain, self.buyer_target, 
                                      fit_by_residual=self.fit_by_residual)