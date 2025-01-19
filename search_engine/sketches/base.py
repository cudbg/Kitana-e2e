import torch
import psutil
from functools import reduce
from ..utils.logging_utils import log_execution, handle_exceptions, logger
from .loader import SketchLoader
import pandas as pd
from itertools import combinations
import logging
from ..config.config import get_config


class SketchBase:
    @handle_exceptions
    @log_execution(logging.INFO)
    def __init__(self, join_key_domain, device='cpu', is_buyer=False):
        """
        Initialize the SketchBase class.

        :param join_key_domain: Dictionary containing the domain of each join key
        :param device: Device type ('cpu' or 'cuda')
        :param is_buyer: Whether it is a buyer
        """
        config = get_config()
        self.device = config.search.device
        self.batch_size = config.search.batch_size
        self.feature_index_mapping = {}
        self.dfid_feature_mapping = {}
        self.device = device
        self.join_key_domain = join_key_domain
        self.current_df_id = 0
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.init()
            gpu_total_mem = torch.cuda.get_device_properties(0).total_memory
            self.gpu_free_mem = gpu_total_mem - torch.cuda.memory_allocated(0)
        else:
            self.gpu_free_mem = None
        if self.batch_size == 'auto':
            self.gpu_batch_size, self.ram_batch_size = self.estimate_batch_size()
        else:
            self.gpu_batch_size = self.ram_batch_size = int(self.batch_size)
        self.sketch_loader = SketchLoader(self.gpu_batch_size, device=self.device, is_buyer=is_buyer)

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def estimate_batch_size(self):
        """
        Estimate the batch size based on the available memory.

        :return: A tuple containing the batch size for GPU and RAM
        """
        bytes_per_element = 4
        tensor_width = reduce(lambda x, y: x * len(y), self.join_key_domain.values(), 1)
        memory = psutil.virtual_memory()
        available_memory = memory.available // 2
        ram_batch_size = available_memory // (bytes_per_element * 3 * tensor_width)
        if not self.gpu_free_mem or not torch.cuda.is_available():
            gpu_batch_size = ram_batch_size
        else:
            gpu_batch_size = self.gpu_free_mem // (bytes_per_element * 3 * tensor_width)
        return gpu_batch_size, ram_batch_size

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def _register_df(self, df_id, feature_num, seller_1, seller_x, seller_x_x, seller_x_y=None, to_disk=False):
        """
        Register a dataframe to the sketch base.

        :param df_id: Unique identifier of the dataframe
        :param feature_num: Number of features in the dataframe
        :param seller_1: 1 matrix of the dataframe
        :param seller_x: X matrix of the dataframe
        :param seller_x_x: X_X matrix of the dataframe
        :param seller_x_y: X_Y matrix of the dataframe
        :param to_disk: Whether to save the sketches to disk
        :return: A dictionary containing the batch_id, df_id, and offset
        """
        if seller_x.size(1) > min(self.gpu_batch_size, self.ram_batch_size):
            raise ValueError("The number of features in the dataframe is too large.")
        self.sketch_loader.load_sketches(
            seller_1=seller_1,
            seller_x=seller_x,
            seller_x_x=seller_x_x,
            seller_x_y=seller_x_y,
            feature_index_map=self.feature_index_mapping,
            seller_id=df_id,
            to_disk=to_disk
        )
        def find_by_seller_id(feature_index_map, seller_id):
            for batch_id, entries in feature_index_map.items():
                for end_pos, id, offset in entries:
                    if id == seller_id:
                        return batch_id, offset
            return None, None

        batch_id, offset = find_by_seller_id(self.feature_index_mapping, df_id)
        return {"batch_id": batch_id, "df_id": df_id, "offset": offset}

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def _calibrate(self, df_id, df, num_features, key_domains, join_keys, normalized=True, fit_by_residual=False, is_buyer=False):
        """
        Calibrate the dataframe to get the 1, X, X_X, and X_Y matrices.

        :param df_id: Unique identifier of the dataframe
        :param df: Dataframe to be calibrated
        :param num_features: Number of features in the dataframe
        :param key_domains: Dictionary containing the domain of each join key
        :param join_keys: List of join keys
        :param normalized: Whether to normalize the data
        :param fit_by_residual: Whether to fit by residual
        :param is_buyer: Whether the dataframe is a buyer
        :return: A tuple containing the calibrated 1, X, X_X, and X_Y tensors
        """
        non_join_key_columns = df.columns.difference(join_keys)
        df_squared = df[non_join_key_columns] ** 2
        df_squared[join_keys] = df[join_keys]

        seller_sum = df.groupby(join_keys).sum()
        ordered_columns = list(seller_sum.columns)

        if df_id not in self.dfid_feature_mapping:
            self.dfid_feature_mapping[df_id] = ordered_columns
        else:
            self.dfid_feature_mapping[df_id] += ordered_columns
        
        seller_sum_squares = df_squared.groupby(join_keys).sum()[ordered_columns]
        seller_count = df.groupby(join_keys).size().to_frame('count')

        if not fit_by_residual and is_buyer:
            df_cross, ordered_cross_cols = {}, []
            for col1, col2 in combinations(ordered_columns, 2):
                df_cross[f"{col1}_{col2}"] = df[col1] * df[col2]
                ordered_cross_cols.append(f"{col1}_{col2}")
            df_cross = pd.DataFrame(df_cross)
            df_cross[join_keys] = df[join_keys]
            seller_sum_cross = df_cross.groupby(join_keys).sum()[ordered_cross_cols]
            if normalized:
                seller_sum_cross = seller_sum_cross.div(seller_count['count'], axis=0)

        if normalized:
            seller_sum = seller_sum.div(seller_count['count'], axis=0)
            seller_sum_squares = seller_sum_squares.div(seller_count['count'], axis=0)
            seller_count = seller_count.assign(count=1)

        if not isinstance(seller_sum.index, pd.MultiIndex):
            seller_sum.index = pd.MultiIndex.from_arrays([seller_sum.index], names=join_keys)
            seller_sum_squares.index = pd.MultiIndex.from_arrays([seller_sum_squares.index], names=join_keys)
            seller_count.index = pd.MultiIndex.from_arrays([seller_count.index], names=join_keys)
            if not fit_by_residual and is_buyer:
                seller_sum_cross.index = pd.MultiIndex.from_arrays([seller_sum_cross.index], names=join_keys)

        index_ranges = [key_domains[col] for col in join_keys]
        multi_index = pd.MultiIndex.from_product(index_ranges, names=join_keys)
        temp_df = pd.DataFrame(index=multi_index)

        seller_x = seller_sum.reindex(multi_index, fill_value=0)
        seller_x = seller_x[seller_x.index.isin(temp_df.index)].values

        seller_x_x = seller_sum_squares.reindex(multi_index, fill_value=0)
        seller_x_x = seller_x_x[seller_x_x.index.isin(temp_df.index)].values

        seller_count = seller_count.reindex(multi_index, fill_value=1)
        seller_count = seller_count[seller_count.index.isin(temp_df.index)].values

        seller_x_y_tensor = None
        if not fit_by_residual and is_buyer:
            seller_x_y = seller_sum_cross.reindex(multi_index, fill_value=0)
            seller_x_y = seller_x_y[seller_x_y.index.isin(temp_df.index)].values
            seller_x_y_tensor = torch.tensor(seller_x_y, dtype=torch.float32)

        seller_x_tensor = torch.tensor(seller_x, dtype=torch.float32)
        seller_x_x_tensor = torch.tensor(seller_x_x, dtype=torch.float32)
        seller_count_tensor = torch.tensor(seller_count, dtype=torch.int).view(-1, 1)
        seller_1_tensor = seller_count_tensor.expand(-1, num_features)

        return seller_x_tensor, seller_x_x_tensor, seller_1_tensor, seller_x_y_tensor

    @handle_exceptions
    @log_execution(logging.INFO)
    def get_df_by_feature_index(self, batch_id, feature_index):
        """
        Get the dataframe ID and feature name by feature index.

        :param batch_id: Batch ID
        :param feature_index: Feature index
        :return: A tuple containing the dataframe ID and feature name
        """
        def bisect(a, x):
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if x < a[mid][0]:
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        index = bisect(self.feature_index_mapping[batch_id], feature_index) - 1
        print("In the get_df_by_feature_index", self.feature_index_mapping[batch_id], feature_index, index)
        start_index, df_id, offset = self.feature_index_mapping[batch_id][index]
        local_feature_index = feature_index - start_index + offset
        print("In the get_df_by_feature_index", self.dfid_feature_mapping[df_id], local_feature_index)
        return df_id, self.dfid_feature_mapping[df_id][local_feature_index]

    @handle_exceptions
    @log_execution(logging.INFO)
    def get_sketch_loader(self):
        """
        Get the sketch loader.

        :return: Sketch loader
        """
        return self.sketch_loader
