class SketchLoader:
    def __init__(self, batch_size, device='cpu', disk_dir='sketches/', is_buyer=False):
        self.batch_size = batch_size
        self.sketch_1_batch = {}
        self.sketch_x_batch = {}
        self.sketch_x_x_batch = {}
        self.sketch_x_y_batch = {}
        self.is_buyer = is_buyer
        self.device = device
        self.num_batches = 0
        self.disk_dir = disk_dir

    def load_sketches(self, seller_1, seller_x, seller_x_x, feature_index_map, seller_id, 
                      cur_df_offset=0, to_disk=False, seller_x_y=None):
        
        if self.is_buyer:
            # Each buyer sketch will only have one column with respect to each join key
            # TODO: Now assume each buyer sketch is small
            if seller_x_y is not None:
                self.sketch_1_batch[0] = seller_1[:, 0:1].to(self.device)
                self.sketch_x_y_batch[0] = seller_x_y.to(self.device)
            else:
                self.sketch_1_batch[0] = seller_1.to(self.device)
            self.sketch_x_batch[0] = seller_x.to(self.device)
            self.sketch_x_x_batch[0] = seller_x_x.to(self.device)
            feature_index_map[0] = [(0, seller_id, 0)]
            return
        
        if not self.sketch_x_batch:
            # If the dictionary is empty, start with batch number 0
            self.sketch_1_batch[0] = seller_1[:, :min(
                self.batch_size, seller_1.size(1))]
            remaining_seller_1 = seller_1[:, self.batch_size:]
            self.sketch_x_batch[0] = seller_x[:, :min(
                self.batch_size, seller_x.size(1))]
            remaining_seller_x = seller_x[:, self.batch_size:]
            self.sketch_x_x_batch[0] = seller_x_x[:, :min(
                self.batch_size, seller_x_x.size(1))]
            remaining_seller_x_x = seller_x_x[:, self.batch_size:]
            feature_index_map[0] = [(0, seller_id, 0)]
            cur_df_offset = self.batch_size
        else:
            # Find the last batch number
            last_batch_num = max(self.sketch_x_batch.keys())
            last_batch_1 = self.sketch_1_batch[last_batch_num]
            last_batch_x = self.sketch_x_batch[last_batch_num]
            last_batch_x_x = self.sketch_x_x_batch[last_batch_num]

            # Calculate remaining space in the last batch
            remaining_space = self.batch_size - last_batch_x.size(1)

            # Append as much as possible to the last batch
            if remaining_space > 0:
                amount_to_append = min(remaining_space, seller_x.size(1))
                self.sketch_1_batch[last_batch_num] = torch.cat(
                    [last_batch_1, seller_1[:, :amount_to_append]], dim=1)
                self.sketch_x_batch[last_batch_num] = torch.cat(
                    [last_batch_x, seller_x[:, :amount_to_append]], dim=1)
                self.sketch_x_x_batch[last_batch_num] = torch.cat(
                    [last_batch_x_x, seller_x_x[:, :amount_to_append]], dim=1)
                remaining_seller_1 = seller_1[:, amount_to_append:]
                remaining_seller_x = seller_x[:, amount_to_append:]
                remaining_seller_x_x = seller_x_x[:, amount_to_append:]
                bisect.insort(feature_index_map[last_batch_num], (last_batch_x.size(
                    1), seller_id, cur_df_offset))
                cur_df_offset += remaining_space
            else:
                # No space left in the last batch, start a new batch
                last_batch_num += 1
                # feature_index_map[last_batch_num] =
                self.sketch_1_batch[last_batch_num] = seller_1[:, :min(
                    self.batch_size, seller_1.size(1))]
                self.sketch_x_batch[last_batch_num] = seller_x[:, :min(
                    self.batch_size, seller_x.size(1))]
                self.sketch_x_x_batch[last_batch_num] = seller_x_x[:, :min(
                    self.batch_size, seller_x_x.size(1))]
                remaining_seller_1 = seller_1[:, self.batch_size:]
                remaining_seller_x = seller_x[:, self.batch_size:]
                remaining_seller_x_x = seller_x_x[:, self.batch_size:]
                feature_index_map[last_batch_num] = [
                    (0, seller_id, cur_df_offset)]
                cur_df_offset += self.batch_size
        self.num_batches = len(self.sketch_x_batch.keys())

        # Recursively append the remaining parts
        # if there is remaining part, that means the previous batch is occupied 
        if remaining_seller_x.size(1) > 0:           
            # Create the directory if it doesn't exist
            if not os.path.exists(self.disk_dir):
                os.makedirs(self.disk_dir)
            # Save the tensor
            if to_disk:
                prev_batch_id = self.num_batches-1
                sketch_1_batch = self.sketch_1_batch[prev_batch_id]
                sketch_x_batch = self.sketch_x_batch[prev_batch_id]
                sketch_x_x_batch = self.sketch_x_x_batch[prev_batch_id]

                torch.save(sketch_1_batch, os.path.join(self.disk_dir, "sketch_1_" + str(prev_batch_id) + ".pt"))
                torch.save(sketch_x_batch, os.path.join(self.disk_dir, "sketch_x_" + str(prev_batch_id) + ".pt"))
                torch.save(sketch_x_x_batch, os.path.join(self.disk_dir, "sketch_x_x_" + str(prev_batch_id) + ".pt"))
                del self.sketch_1_batch[prev_batch_id]
                del self.sketch_x_batch[prev_batch_id]
                del self.sketch_x_x_batch[prev_batch_id]
            self.load_sketches(remaining_seller_1, remaining_seller_x, remaining_seller_x_x,
                               feature_index_map, seller_id, cur_df_offset) 
            
    def get_sketches(self, batch_id, from_disk=False):
        sketch_x_y_batch = None
        if from_disk:
            # Buyer dataset never on disk
            sketch_1_batch = torch.load(os.path.join(self.disk_dir, 
                                                     "sketch_1_" + str(batch_id) + ".pt")).to(self.device)
            sketch_x_batch = torch.load(os.path.join(self.disk_dir, 
                                                     "sketch_x_" + str(batch_id) + ".pt")).to(self.device)
            sketch_x_x_batch = torch.load(os.path.join(self.disk_dir, 
                                                       "sketch_x_x_" + str(batch_id) + ".pt")).to(self.device)
        else:
            sketch_1_batch = self.sketch_1_batch[batch_id].to(self.device)
            sketch_x_batch = self.sketch_x_batch[batch_id].to(self.device)
            sketch_x_x_batch = self.sketch_x_x_batch[batch_id].to(self.device)
            if batch_id in self.sketch_x_y_batch:
                sketch_x_y_batch = self.sketch_x_y_batch[batch_id].to(self.device)
        return sketch_1_batch, sketch_x_batch, sketch_x_x_batch, sketch_x_y_batch

    def get_num_batches(self):
        return self.num_batches
    
class SketchBase:
    def __init__(self, join_key_domain, device='cpu', is_buyer=False):
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
        self.gpu_batch_size, self.ram_batch_size = self.estimate_batch_size()    
        # TODO: join key domain requried in estimate_batch_size
        # sketch loader only needs to fully utilize gpu memory
        self.sketch_loader = SketchLoader(self.gpu_batch_size, device=device, is_buyer=is_buyer)


    """
    This function is used to estimate the batch size based on the available memory.
    It will return the batch size for both GPU and RAM.

    @param join_key_domain: a dictionary containing the domain of each join key. 
            We need this because the size(0) would be the product of the domain of all join keys.
    
    @return: a tuple containing the batch size for GPU and RAM

    """
    def estimate_batch_size(self):
        # Similar logic as search_gpu.py, just copy and paste
        bytes_per_element = 4
        tensor_width = reduce(lambda x, y: x * len(y), 
                              self.join_key_domain.values(), 1)
        memory = psutil.virtual_memory()
        # TODO: 2 is a workaround
        available_memory = memory.available // 2
        ram_batch_size = available_memory // (bytes_per_element * 3 * tensor_width)
        if not self.gpu_free_mem or not torch.cuda.is_available():
            gpu_batch_size = ram_batch_size
        else:
            gpu_batch_size = self.gpu_free_mem // (bytes_per_element * 3 * tensor_width)
        return gpu_batch_size, ram_batch_size

    """
    This function is used to register a dataframe to the sketch base.
    As a base function, it only takes in 1, x, x_x, feature_num, df_id, and offset as input params. 
    It will check if the current tensors satisfies the ram requirements and load it to sketches with load_sketches. 
    After loading the sketches, it returns the updated offset corresponding to the id of the dataframe.

    @param df_id: the unique identifier of the dataframe. Later on, when we want to fetch the 
              corresponding df_id, we could use this identifier. It stores in a priority queue in feature_index_mapping.
    @param feature_num: the number of features in the dataframe
    @param seller_1: the 1 matrix of the dataframe
    @param seller_x: the X matrix of the dataframe
    @param seller_x_x: the X_X matrix of the dataframe
    @param to_disk: whether to save the sketches to disk

    @return: a dictionary containing the batch_id, df_id, and offset
    """
    def _register_df(self, df_id, feature_num, seller_1, seller_x, seller_x_x, seller_x_y=None, to_disk=False):
        # Before loading the sketches, check if the current tensors satisfy the ram requirements
        if seller_x.size(1) > min(self.gpu_batch_size, self.ram_batch_size):
            raise ValueError("The number of features in the dataframe is too large.")
        # Load the sketches
        self.sketch_loader.load_sketches(
            seller_1 = seller_1,
            seller_x = seller_x,
            seller_x_x = seller_x_x,
            seller_x_y = seller_x_y,
            feature_index_map = self.feature_index_mapping,
            seller_id = df_id,
            to_disk = to_disk
        )
        # Return the updated offset corresponding to the df_id. This is not efficient. Only for unit test usage.
        def find_by_seller_id(feature_index_map, seller_id):
            for batch_id, entries in feature_index_map.items():
                for end_pos, id, offset in entries:
                    if id == seller_id:
                        return batch_id, offset
            return None, None  # If the seller_id is not found

        batch_id, offset = find_by_seller_id(self.feature_index_mapping, df_id)
        return {"batch_id": batch_id, "df_id": df_id, "offset": offset}



    """
    This function is used to get the 1, X, X_X, and X_Y matrices of a dataframe.
    takes in a df( with join key as a col)  and the key_domainsreturn and returns the calibrated 1, x, x_x tensors.
    @param df_id: the unique identifier of the dataframe.
    @param df: the dataframe to be calibrated. This df should be with join keys.
    @param num_features: the number of features in the dataframe.
    @param key_domains: a dictionary containing the domain of each join key.
    @param join_keys: a list containing the names of the join keys.
    @param fit_by_residual: a boolean indicating whether to fit by residual.
    @param is_buyer: a boolean indicating whether the df is a buyer or seller.

    @return: a tuple containing the calibrated 1, x, x_x tensors
    """

    def _calibrate(self, df_id, df, num_features, key_domains, join_keys, normalized=True, fit_by_residual=False, is_buyer=False):
        # Get a squared df but not include join keys
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
        
                
        # Normalize by seller_count if normalization is enabled
        if normalized:
            seller_sum = seller_sum.div(seller_count['count'], axis=0)
            seller_sum_squares = seller_sum_squares.div(seller_count['count'], axis=0)

            # Set seller_count to 1 for each group
            seller_count = seller_count.assign(count=1)

        if not isinstance(seller_sum.index, pd.MultiIndex):
            seller_sum.index = pd.MultiIndex.from_arrays(
                [seller_sum.index], names=join_keys)
            seller_sum_squares.index = pd.MultiIndex.from_arrays(
                [seller_sum_squares.index], names=join_keys)
            seller_count.index = pd.MultiIndex.from_arrays(
                [seller_count.index], names=join_keys)
            
            if not fit_by_residual and is_buyer:
                seller_sum_cross.index = pd.MultiIndex.from_arrays(
                    [seller_sum_cross.index], names=join_keys)
            
        # Create the correct multi_index for cartesian product
        index_ranges = [key_domains[col] for col in join_keys]
        multi_index = pd.MultiIndex.from_product(index_ranges, names=join_keys)
        
        # Temporary DataFrame to facilitate 'inner' join
        temp_df = pd.DataFrame(index=multi_index)

        # Reindex and perform inner join
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
        
        # Convert to PyTorch tensors
        seller_x_tensor = torch.tensor(seller_x, dtype=torch.float32)
        seller_x_x_tensor = torch.tensor(seller_x_x, dtype=torch.float32)
        seller_count_tensor = torch.tensor(
            seller_count, dtype=torch.int).view(-1, 1)
        seller_1_tensor = seller_count_tensor.expand(-1, num_features)

        return seller_x_tensor, seller_x_x_tensor, seller_1_tensor, seller_x_y_tensor
    
    """
    This function gets the batch_id and the feature_index in this batch. These are all found in the searchEngine class.
    @param batch_id: the batch_id of the feature_index
    @param feature_index: the feature_index of the feature in the batch

    @return: a tuple containing the df_id and a feature name indicated by the dfid_feature_mapping
    """
    def get_df_by_feature_index(self, batch_id, feature_index):
        # Perform a binary search to find the right interval
        # bisect.bisect returns the insertion point which gives us the index where the feature_index would be inserted to maintain order.
        # We subtract one to get the tuple corresponding to the start index of the range that the feature_index falls into.
        def bisect(a, x):
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if x < a[mid][0]:  # Compare with the first element of the tuple at mid
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        index = bisect(self.feature_index_mapping[batch_id], feature_index) - 1
        start_index, df_id, offset = self.feature_index_mapping[batch_id][index]
        # Calculate the local feature index within the seller's dataset
        local_feature_index = feature_index - start_index + offset
        return df_id, self.dfid_feature_mapping[df_id][local_feature_index]
    
    def get_sketch_loader(self):
        return self.sketch_loader
    
"""
This class wraps up a seller df to a sketch. It will be used to register the seller df to the sketch base 
and store the corresponding batch_id and offset. It also stores the related information such as join keys,
join key domains, and the sketch base object.
"""
class SellerSketch():
    def __init__(self, seller_df: pd.DataFrame, join_keys: list, join_key_domains: dict, sketch_base: SketchBase, df_id: int, device='cpu'):
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


    """
    This function is used to register a seller df to the sketch base.

    @return: a tuple containing the batch_id and offset
    """
    def register_this_seller(self):
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
    
    def get_base(self):
        return self.sketch_base
    
    def get_sketches(self):
        return self.sketch_base.sketch_loader.get_sketches(self.batch_id)
    
    def get_df(self):
        return self.seller_df

"""
This class wraps up a buyer df to a sketch. It will be used to register the buyer df to the sketch base
and store the corresponding batch_id and offset. It also stores the related information such as join keys,
join key domains, and the sketch base object. 

One more thing it stores is the target feature and the corresponding index of the target feature in the buyer df.
"""
class BuyerSketch():
    def __init__(self, buyer_df: pd.DataFrame, join_keys: list, join_key_domains: dict, sketch_base: SketchBase, target_feature: str, device='cpu', fit_by_residual=False):
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


    """
    This function is used to register a buyer df to the sketch base. 

    @return: a tuple containing the batch_id and offset
    """
    def register_this_buyer(self, fit_by_residual=False):
        # Calibrate the df
        buyer_x, buyer_x_x, buyer_1, buyer_x_y = self.sketch_base._calibrate(
            self.df_id, self.buyer_df, len(self.buyer_df.columns) - len(self.join_keys), self.join_key_domains, self.join_keys, is_buyer=True, fit_by_residual=fit_by_residual)
        # Register the df
        result = self.sketch_base._register_df(df_id= self.df_id, feature_num=len(self.buyer_df.columns) - len(self.join_keys), seller_1=buyer_1, seller_x=buyer_x, seller_x_x=buyer_x_x, seller_x_y=buyer_x_y)
        self.batch_id = result["batch_id"]
        self.offset = result["offset"]
        # We don't update df_id here because it is the id of the buyer_df

        return self.batch_id, self.offset
    
    def get_base(self):
        return self.sketch_base
    
    """
    This function gets the 1, X, X_X, and X_Y matrices of the buyer df, which is stored in the sketch base.

    @return: a tuple containing the 1, X, X_X, and X_Y matrices
    """
    def get_sketches(self):
        return self.sketch_base.sketch_loader.get_sketches(self.batch_id)
    
    """
    This function gets the feature index and name of the target feature in the buyer df.

    @return: a dictionary containing the index and name of the target feature
    """
    def get_target_feature(self):
        return {"index": self.target_feature_index, "name": self.target_feature}