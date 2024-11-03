# Import necessary modules
from .buyer_data import PrepareBuyer
from .sellers import PrepareSellers
from .seller_data import PrepareSeller
from ..utils.logging_utils import log_execution, handle_exceptions
import logging

class PrepareBuyerSellers():
    @log_execution(logging.INFO)
    def __init__(self, need_to_clean_data: bool = True):
        """
        Initializes a management system for handling both buyers and sellers with data cleaning options.
        """
        self.buyer = None
        self.buyer_added = False
        self.need_to_clean_data = need_to_clean_data
        self.sellers = PrepareSellers()

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def filter_join_keys(self, join_keys):
        """
        Filters and validates join keys based on seller domain keys.
        """
        domain_dict = self.sellers.join_key_domains
        valid_join_keys = []
        join_key_changed = False
    
        for keys in join_keys:
            key = '_'.join(keys) if len(keys) > 1 else keys[0]
            if key in domain_dict:
                valid_join_keys.append(keys)

        return valid_join_keys, join_key_changed
    
    @handle_exceptions
    @log_execution(logging.DEBUG)
    def get_buyer(self):
        """
        Returns the buyer object.
        """
        return self.buyer
    
    @handle_exceptions
    @log_execution(logging.DEBUG)
    def get_sellers(self):
        """
        Returns the sellers object.
        """
        return self.sellers
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def add_buyer(self, buyer: PrepareBuyer):
        """
        Adds a buyer to the system and sets the buyer_added flag.
        """
        self.buyer_added = True
        self.buyer = buyer
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def add_seller(self, seller_name: str, seller: PrepareSeller):
        """
        Adds a seller to the system after ensuring a buyer has been added.
        """
        if not self.buyer_added:
            raise Exception("Buyer has not been added yet")
        self.sellers.add_sellers(seller_name=seller_name, seller=seller, buyer=self.buyer)
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def add_buyer_by_path(self, data_path: str, join_keys: list, buyer_features: list, target_feature: str):
        """
        Adds a buyer by specifying a data path and related details.
        """
        self.add_buyer(PrepareBuyer(data_path, join_keys, target_feature, features=buyer_features, need_to_clean_data=self.need_to_clean_data))
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def add_seller_by_path(self, data_path: str, join_keys: list, seller_features: list):
        """
        Adds a seller by specifying a data path and related details.
        """
        self.sellers.add_seller_by_path(data_path, join_keys, self.buyer, features=seller_features, need_to_clean_data=self.need_to_clean_data)
