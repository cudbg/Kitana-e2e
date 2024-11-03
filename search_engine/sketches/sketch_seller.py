import pandas as pd
from ..utils.logging_utils import log_execution, handle_exceptions
from .base import SketchBase
import logging

class SellerSketch:
    @handle_exceptions
    @log_execution(logging.INFO)
    def __init__(self, seller_df: pd.DataFrame, join_keys: list, join_key_domains: dict, sketch_base: SketchBase, df_id: int, device='cpu'):
        """
        Initialize the SellerSketch class.

        :param seller_df: Seller dataframe
        :param join_keys: List of join keys
        :param join_key_domains: Dictionary containing the domain of each join key
        :param sketch_base: Sketch base object
        :param df_id: Dataframe ID
        :param device: Device type ('cpu' or 'cuda')
        """
        self.join_keys = join_keys
        self.join_key_domains = join_key_domains
        self.all_join_keys = [key for key in self.join_key_domains.keys()]
        self.device = device
        self.df_id = df_id

        # Seller's dataframe will be stored in this variable
        self.seller_df = seller_df

        # This is from the sketch base. Will be updated after registering the seller sketch
        self.batch_id = 0
        self.offset = 0

        # This stores a seller sketch base. This single seller df will use the sketch base to register itself
        self.sketch_base = sketch_base

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def register_this_seller(self):
        """
        Register the seller dataframe to the sketch base.

        :return: A tuple containing the batch_id and offset
        """
        # First we should cut the df into partitions to maximize the GPU and RAM usage
        ram_batch_size = self.sketch_base.ram_batch_size
        # Rename the columns and add the join keys as prefix to the column names
        prefix = "_".join(self.join_keys) + "_"
        self.seller_df.columns = [prefix + col if col not in self.join_keys else col for col in self.seller_df.columns]
        feature_columns = [col for col in self.seller_df.columns if col not in self.join_keys]
        if len(self.seller_df.columns) > ram_batch_size:
            features_per_partition = ram_batch_size - 1
            # Splitting the DataFrame into partitions
            num_partitions = (len(feature_columns) // features_per_partition) + (len(feature_columns) % features_per_partition > 0)
            for i in range(num_partitions):
                cur_features = feature_columns[i * features_per_partition:(i + 1) * features_per_partition] # Avoid address coding
                cols = self.join_keys + cur_features
                # Creating a new DataFrame for this partition
                cur_df = self.seller_df[cols]
                # Calibrate the df
                seller_x, seller_x_x, seller_1, seller_x_y = self.sketch_base._calibrate(
                    self.df_id, cur_df, len(cur_features), self.join_key_domains, self.join_keys)
                # Register the df
                result = self.sketch_base._register_df(self.df_id, len(cur_features), seller_1, seller_x, seller_x_x)
                self.batch_id = result["batch_id"]
                self.offset = result["offset"]
        else:
            # Directly calibrate the df
            seller_x, seller_x_x, seller_1, seller_x_y = self.sketch_base._calibrate(
                self.df_id, self.seller_df, len(self.seller_df.columns) - len(self.join_keys), self.join_key_domains, self.join_keys)
            # Register the df
            result = self.sketch_base._register_df(self.df_id, len(self.seller_df.columns) - len(self.join_keys), seller_1, seller_x, seller_x_x)
            self.batch_id = result["batch_id"]
            self.offset = result["offset"]
            # We don't update df_id here because it is the id of the seller_df

        return self.batch_id, self.offset

    @handle_exceptions
    @log_execution(logging.INFO)
    def get_base(self):
        """
        Get the sketch base.

        :return: Sketch base
        """
        return self.sketch_base

    @handle_exceptions
    @log_execution(logging.INFO)
    def get_sketches(self):
        """
        Get the sketches.

        :return: Sketches
        """
        return self.sketch_base.sketch_loader.get_sketches(self.batch_id)

    @handle_exceptions
    @log_execution(logging.INFO)
    def get_df(self):
        """
        Get the seller dataframe.

        :return: Seller dataframe
        """
        return self.seller_df
