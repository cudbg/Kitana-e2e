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
        # Initialize the list to hold the validated join keys
        valid_join_keys = []
        join_key_changed = False
    
        # Iterate over each list of keys in the join_keys list
        for keys in join_keys:
            # Form the key as it would appear in the domain dictionary
            # Join the keys with an underscore if there's more than one key in the list
            if len(keys) > 1:
                key = '_'.join(keys)
            else:
                key = keys[0]
            # Check if the formed key is in the domain dictionary
            if key in domain_dict:
                # If the key exists, add the original list of keys to the valid list
                valid_join_keys.append(key)

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
    def get_buyer_data(self):
        """
        Returns the buyer data.
        """
        return self.buyer.get_data()

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def get_buyer_join_keys(self):
        """
        Returns the buyer join keys.
        """
        return self.buyer.get_join_keys()

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def get_sellers(self):
        """
        Returns the sellers object.
        """
        return self.sellers
    
    @handle_exceptions
    @log_execution(logging.DEBUG)
    def get_seller_data(self):
        """
        Returns the seller data.
        """
        return self.sellers.get_sellers()

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

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def get_domain(self):
        """
        Returns the domain of the join keys.
        """
        return self.sellers.join_key_domains