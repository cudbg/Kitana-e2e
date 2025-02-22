import os
import torch
import bisect
from ..utils.logging_utils import log_execution, handle_exceptions, logger
import logging

class SketchLoader:
    @handle_exceptions
    @log_execution(logging.INFO)
    def __init__(self, batch_size, device='cpu', disk_dir='sketches/', is_buyer=False):
        """
        Initialize the SketchLoader class.

        :param batch_size: Batch size
        :param device: Device type ('cpu' or 'cuda')
        :param disk_dir: Disk directory
        :param is_buyer: Whether it is a buyer
        """
        self.batch_size = batch_size
        self.sketch_1_batch = {}
        self.sketch_x_batch = {}
        self.sketch_x_x_batch = {}
        self.sketch_x_y_batch = {}
        self.is_buyer = is_buyer
        self.device = device
        self.num_batches = 0
        self.disk_dir = disk_dir

    @handle_exceptions
    @log_execution(logging.DEBUG)
    def load_sketches(self, seller_1, seller_x, seller_x_x, feature_index_map, seller_id, 
                      cur_df_offset=0, to_disk=False, seller_x_y=None):
        """
        Load sketch data.

        :param seller_1: Seller 1 matrix
        :param seller_x: Seller X matrix
        :param seller_x_x: Seller X_X matrix
        :param feature_index_map: Feature index map
        :param seller_id: Seller ID
        :param cur_df_offset: Current dataframe offset
        :param to_disk: Whether to save to disk
        :param seller_x_y: Seller X_Y matrix
        """
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
            self.sketch_1_batch[0] = seller_1[:, :min(self.batch_size, seller_1.size(1))]
            remaining_seller_1 = seller_1[:, self.batch_size:]
            self.sketch_x_batch[0] = seller_x[:, :min(self.batch_size, seller_x.size(1))]
            remaining_seller_x = seller_x[:, self.batch_size:]
            self.sketch_x_x_batch[0] = seller_x_x[:, :min(self.batch_size, seller_x_x.size(1))]
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
                bisect.insort(feature_index_map[last_batch_num], (last_batch_x.size(1), seller_id, cur_df_offset))
                cur_df_offset += remaining_space
            else:
                # No space left in the last batch, start a new batch
                last_batch_num += 1
                self.sketch_1_batch[last_batch_num] = seller_1[:, :min(self.batch_size, seller_1.size(1))]
                self.sketch_x_batch[last_batch_num] = seller_x[:, :min(self.batch_size, seller_x.size(1))]
                self.sketch_x_x_batch[last_batch_num] = seller_x_x[:, :min(self.batch_size, seller_x_x.size(1))]
                remaining_seller_1 = seller_1[:, self.batch_size:]
                remaining_seller_x = seller_x[:, self.batch_size:]
                remaining_seller_x_x = seller_x_x[:, self.batch_size:]
                feature_index_map[last_batch_num] = [(0, seller_id, cur_df_offset)]
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
                prev_batch_id = self.num_batches - 1
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
                               feature_index_map, seller_id, cur_df_offset, to_disk)
        if to_disk:
            # save feature_index_map
            torch.save(feature_index_map, os.path.join(self.disk_dir, "feature_index_map.pt"))


    @handle_exceptions
    @log_execution(logging.INFO)
    def get_sketches(self, batch_id, from_disk=False):
        """
        Get sketch data.

        :param batch_id: Batch ID
        :param from_disk: Whether to load from disk
        :return: Sketch data
        """
        sketch_x_y_batch = None
        if from_disk:
            # Buyer dataset never on disk
            sketch_1_batch = torch.load(os.path.join(self.disk_dir, "sketch_1_" + str(batch_id) + ".pt")).to(self.device)
            sketch_x_batch = torch.load(os.path.join(self.disk_dir, "sketch_x_" + str(batch_id) + ".pt")).to(self.device)
            sketch_x_x_batch = torch.load(os.path.join(self.disk_dir, "sketch_x_x_" + str(batch_id) + ".pt")).to(self.device)
        else:
            sketch_1_batch = self.sketch_1_batch[batch_id].to(self.device)
            sketch_x_batch = self.sketch_x_batch[batch_id].to(self.device)
            sketch_x_x_batch = self.sketch_x_x_batch[batch_id].to(self.device)
            if batch_id in self.sketch_x_y_batch:
                sketch_x_y_batch = self.sketch_x_y_batch[batch_id].to(self.device)
        return sketch_1_batch, sketch_x_batch, sketch_x_x_batch, sketch_x_y_batch

    @handle_exceptions
    @log_execution(logging.INFO)
    def get_feature_index_map(self, from_disk=False):
        """
        Get the feature index map.

        :param batch_id: Batch ID
        :param from_disk: Whether to load from disk
        :return: Feature index map
        """
        if from_disk:
            return torch.load(os.path.join(self.disk_dir, "feature_index_map.pt"))
        return self.feature_index_map

    @handle_exceptions
    @log_execution(logging.INFO)
    def get_num_batches(self):
        """
        Get the number of batches.

        :return: Number of batches
        """
        return self.num_batches
