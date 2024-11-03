import pandas as pd
from ..utils.logging_utils import log_execution, handle_exceptions
from .base import SketchBase
import logging

class BuyerSketch:
    @handle_exceptions
    @log_execution(logging.INFO)
    def __init__(self, buyer_df: pd.DataFrame, join_keys: list, join_key_domains: dict, sketch_base: SketchBase, target_feature: str, device='cpu', fit_by_residual=False):
        """
        Initialize the BuyerSketch class.

        :param buyer_df: Buyer dataframe
        :param join_keys: List of join keys
        :param join_key_domains: Dictionary containing the domain of each join key
        :param sketch_base: Sketch base object
        :param target_feature: Target feature
        :param device: Device type ('cpu' or 'cuda')
        :param fit_by_residual: Whether to fit by residual
        """
        self.join_keys = join_keys
        self.join_key_domains = join_key_domains
        self.device = device
        self.df_id = 0

        # This is to indicate the target feature of the buyer
        self.target_feature = target_feature
        if not fit_by_residual:
            # When fitting by residual, the target feature is not in the buyer_df
            self.target_feature_index = buyer_df.columns.get_loc(target_feature)

        # Buyer's dataframe will be stored in this variable
        self.buyer_df = buyer_df

        # This is from the sketch base. Will be updated after registering the buyer sketch
        self.batch_id = 0 # Since for now the buyer dataset has little chance to exceed the batch size, 
                            # we set the batch_id to 0. Also since the buyer dataset is small, 
                            # we don't need to split it into partitions. So it is not a list.
        self.offset = 0

        # This stores a buyer sketch base. This single buyer df will use the sketch base to register itself
        self.sketch_base = sketch_base

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def register_this_buyer(self, fit_by_residual=False):
        """
        Register the buyer dataframe to the sketch base.

        :param fit_by_residual: Whether to fit by residual
        :return: A tuple containing the batch_id and offset
        """
        # Calibrate the df
        buyer_x, buyer_x_x, buyer_1, buyer_x_y = self.sketch_base._calibrate(
            self.df_id, self.buyer_df, len(self.buyer_df.columns) - len(self.join_keys), self.join_key_domains, self.join_keys, is_buyer=True, fit_by_residual=fit_by_residual)
        # Register the df
        result = self.sketch_base._register_df(df_id= self.df_id, feature_num=len(self.buyer_df.columns) - len(self.join_keys), seller_1=buyer_1, seller_x=buyer_x, seller_x_x=buyer_x_x, seller_x_y=buyer_x_y)
        self.batch_id = result["batch_id"]
        self.offset = result["offset"]
        # We don't update df_id here because it is the id of the buyer_df

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
    def get_target_feature(self):
        """
        Get the feature index and name of the target feature in the buyer dataframe.

        :return: A dictionary containing the index and name of the target feature
        """
        return {"index": self.target_feature_index, "name": self.target_feature}
