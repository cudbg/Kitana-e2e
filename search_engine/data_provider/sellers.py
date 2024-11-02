from .seller_data import PrepareSeller
from .buyer_data import PrepareBuyer
from utils.logging_utils import log_execution, handle_exceptions
import logging

class PrepareSellers():
    @log_execution(logging.INFO)
    def __init__(self):
        """
        Initializes the PrepareSellers class to manage multiple seller datasets.
        """
        self.sellers = {}
        self.join_keys = []
        self.join_key_domains = {}  # {join_key: {set of values}}

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def get_sellers(self):
        """
        Returns the dictionary of sellers.
        """
        return self.sellers
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def see_sellers(self):
        """
        Displays information about all sellers.
        """
        print("Sellers:")
        for seller_name in self.sellers:
            print(f"Seller: {seller_name}")
            self.sellers[seller_name].see_data()
    
    @handle_exceptions
    @log_execution(logging.WARNING)
    def add_sellers(self, seller_name: str, seller: PrepareSeller, buyer: PrepareBuyer):
        """
        Attempts to add a seller to the collection if the seller's data has intersections with the buyer's data.
        """
        if seller.has_key:
            for join_key_in_string in seller.get_join_keys():
                seller_keys_set = set(seller.data[join_key_in_string])
                buyer_keys_set = set(buyer.buyer_key_domain[join_key_in_string])

                intersection = seller_keys_set.intersection(buyer_keys_set)
                if not intersection:
                    print(f"Seller: {seller_name}'s join key: {join_key_in_string} does not have any intersection with the buyer's join key")
                    index = seller.join_keys_in_string.index(join_key_in_string)
                    seller.join_keys_in_string.pop(index)
                    seller.data.drop(join_key_in_string, axis=1, inplace=True)

            if seller.join_keys_in_string == []:
                seller.has_key = False
            else:
                self.sellers[seller_name] = seller
                self.join_keys = list(set(self.join_keys).union(set(seller.get_join_keys())))
                self.update_domain(seller)
                return True
        print(f"Seller: {seller_name} does not have the corresponding join keys")
        return False

    @handle_exceptions
    @log_execution(logging.INFO)
    def add_seller_by_path(self, data_path: str, join_keys: list, buyer: PrepareBuyer, features: list, need_to_clean_data: bool = True):
        """
        Adds a seller by specifying a data path and related information.
        """
        seller_name = data_path.split("/")[-1].split(".")[0]
        self.add_sellers(seller_name, PrepareSeller(data_path, join_keys, features=features, need_to_clean_data=need_to_clean_data), buyer)
    
    @handle_exceptions
    @log_execution(logging.DEBUG)
    def get_domain(self, join_key: str):
        """
        Retrieves the domain for a specific join key.
        """
        return self.join_key_domains[join_key]
    
    @handle_exceptions
    @log_execution(logging.INFO)
    def update_domain(self, seller: PrepareSeller):
        """
        Updates domain information for a seller.
        """
        seller_join_key_domains = seller.record_join_key_domains()
        for join_key in seller_join_key_domains:
            if join_key in self.join_key_domains:
                self.join_key_domains[join_key] = self.join_key_domains[join_key].union(seller_join_key_domains[join_key])
            else:
                self.join_key_domains[join_key] = seller_join_key_domains[join_key]

        for seller_name in self.sellers:
            seller = self.sellers[seller_name]
            seller_join_keys = seller.get_join_keys()
            for join_key in seller_join_keys:
                seller.set_join_key_domains(join_key, self.get_domain(join_key))

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def get_seller_names(self):
        """
        Returns a list of seller names.
        """
        return list(self.sellers.keys())
