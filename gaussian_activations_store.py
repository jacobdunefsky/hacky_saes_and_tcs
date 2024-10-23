import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

import math

import gc

class GaussianActivationsStore:
    """
    Class for generating and storing activations for transcoders
    from Gaussian-distributed MLP input activations
    """
    def __init__(
        self, cfg, model: HookedTransformer, create_dataloader: bool = True,
    ):
        self.cfg = cfg
        self.model = model
        
        if create_dataloader:
            # fill buffer half a buffer, so we can mix it with a new buffer
            self.storage_buffer_out = None
            if self.cfg.is_transcoder:
                # if we're a transcoder, then we want to keep a buffer for our input activations and our output activations
                self.storage_buffer, self.storage_buffer_out = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
            else:
                self.storage_buffer = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
            self.dataloader = self.get_data_loader()

    def get_activations(self):
        assert(self.cfg.hook_point_head_index is None)

        # ASSUMPTION: out_hook_point is mlp_out
        layer = self.cfg.hook_point_layer
        d_in = self.cfg.d_in

        batch_size = self.cfg.store_batch_size
        context_size = self.cfg.context_size
        with torch.no_grad():
            normal_activs = torch.randn(batch_size, context_size, d_in).cuda()
            s = math.sqrt(d_in)
            spherical_activs = torch.einsum('bcd, bc -> bcd', normal_activs, s/torch.norm(normal_activs, dim=-1))
            mlp_out_activs = self.model.blocks[layer].mlp(spherical_activs)
    
        if not self.cfg.is_transcoder:
            activations = mlp_out_activs
        else:
            activations = (spherical_activs, mlp_out_activs)

        return activations

    def get_buffer(self, n_batches_in_buffer):
        gc.collect()
        torch.cuda.empty_cache()
        
        context_size = self.cfg.context_size
        batch_size = self.cfg.store_batch_size
        d_in = self.cfg.d_in
        total_size = batch_size * n_batches_in_buffer

        # TODO: get transcoders working with cached activations
        assert(not (self.cfg.is_transcoder and self.cfg.use_cached_activations))
        if self.cfg.use_cached_activations:
            # Load the activations from disk
            buffer_size = total_size * context_size
            # Initialize an empty tensor (flattened along all dims except d_in)
            new_buffer = torch.zeros((buffer_size, d_in), dtype=self.cfg.dtype,
                                     device=self.cfg.device)
            n_tokens_filled = 0
            
            # The activations may be split across multiple files,
            # Or we might only want a subset of one file (depending on the sizes)
            while n_tokens_filled < buffer_size:
                # Load the next file
                # Make sure it exists
                if not os.path.exists(f"{self.cfg.cached_activations_path}/{self.next_cache_idx}.pt"):
                    print("\n\nWarning: Ran out of cached activation files earlier than expected.")
                    print(f"Expected to have {buffer_size} activations, but only found {n_tokens_filled}.")
                    if buffer_size % self.cfg.total_training_tokens != 0:
                        print("This might just be a rounding error — your batch_size * n_batches_in_buffer * context_size is not divisible by your total_training_tokens")
                    print(f"Returning a buffer of size {n_tokens_filled} instead.")
                    print("\n\n")
                    new_buffer = new_buffer[:n_tokens_filled]
                    break
                activations = torch.load(f"{self.cfg.cached_activations_path}/{self.next_cache_idx}.pt")
                
                # If we only want a subset of the file, take it
                taking_subset_of_file = False
                if n_tokens_filled + activations.shape[0] > buffer_size:
                    activations = activations[:buffer_size - n_tokens_filled]
                    taking_subset_of_file = True
                
                # Add it to the buffer
                new_buffer[n_tokens_filled : n_tokens_filled + activations.shape[0]] = activations
                
                # Update counters
                n_tokens_filled += activations.shape[0]
                if taking_subset_of_file:
                    self.next_idx_within_buffer = activations.shape[0]
                else:
                    self.next_cache_idx += 1
                    self.next_idx_within_buffer = 0
                
            return new_buffer

        refill_iterator = range(0, batch_size * n_batches_in_buffer, batch_size)
        # refill_iterator = tqdm(refill_iterator, desc="generate activations")

        # Initialize empty tensor buffer of the maximum required size
        new_buffer = torch.zeros(
            (total_size, context_size, d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )

        new_buffer_out = None
        if self.cfg.is_transcoder:
            new_buffer_out = torch.zeros(
                (total_size, context_size, self.cfg.d_out),
                dtype=self.cfg.dtype,
                device=self.cfg.device,
            )

        # Insert activations directly into pre-allocated buffer
        # pbar = tqdm(total=n_batches_in_buffer, desc="Filling buffer")
        for refill_batch_idx_start in refill_iterator:
            if not self.cfg.is_transcoder:
                refill_activations = self.get_activations()
                new_buffer[
                    refill_batch_idx_start : refill_batch_idx_start + batch_size
                ] = refill_activations
            else:
                refill_activations_in, refill_activations_out = self.get_activations()
                new_buffer[
                    refill_batch_idx_start : refill_batch_idx_start + batch_size
                ] = refill_activations_in

                new_buffer_out[
                    refill_batch_idx_start : refill_batch_idx_start + batch_size
                ] = refill_activations_out
            
            # pbar.update(1)

        new_buffer = new_buffer.reshape(-1, d_in)
        randperm = torch.randperm(new_buffer.shape[0])
        new_buffer = new_buffer[randperm]

        if self.cfg.is_transcoder:
            new_buffer_out = new_buffer_out.reshape(-1, self.cfg.d_out)
            new_buffer_out = new_buffer_out[randperm]

        if self.cfg.is_transcoder:
            return new_buffer, new_buffer_out
        else:
            return new_buffer

    def get_data_loader(self,) -> DataLoader:
        '''
        Return a torch.utils.dataloader which you can get batches from.
        
        Should automatically refill the buffer when it gets to n % full. 
        (better mixing if you refill and shuffle regularly).
        
        '''
        
        batch_size = self.cfg.train_batch_size
        
        if self.cfg.is_transcoder:
            # ugly code duplication if we're a transcoder
            new_buffer, new_buffer_out = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
            mixing_buffer = torch.cat(
                [new_buffer,
                 self.storage_buffer]
            )
            mixing_buffer_out = torch.cat(
                [new_buffer_out,
                 self.storage_buffer_out]
            )

            assert(mixing_buffer.shape[0] == mixing_buffer_out.shape[0])
            randperm = torch.randperm(mixing_buffer.shape[0])
            mixing_buffer = mixing_buffer[randperm]
            mixing_buffer_out = mixing_buffer_out[randperm]

            self.storage_buffer = mixing_buffer[:mixing_buffer.shape[0]//2]
            self.storage_buffer_out = mixing_buffer_out[:mixing_buffer_out.shape[0]//2]

            # have to properly stack both of our new buffers into the dataloader
            """stacked_buffers = torch.stack([
                mixing_buffer[mixing_buffer.shape[0]//2:],
                mixing_buffer_out[mixing_buffer.shape[0]//2:]
            ], dim=1)"""
            catted_buffers = torch.cat([
                mixing_buffer[mixing_buffer.shape[0]//2:],
                mixing_buffer_out[mixing_buffer.shape[0]//2:]
            ], dim=1)

            #dataloader = iter(DataLoader(stacked_buffers, batch_size=batch_size, shuffle=True))
            dataloader = iter(DataLoader(catted_buffers, batch_size=batch_size, shuffle=True))
        else:
            # 1. # create new buffer by mixing stored and new buffer
            mixing_buffer = torch.cat(
                [self.get_buffer(self.cfg.n_batches_in_buffer // 2),
                 self.storage_buffer]
            )
            
            mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]
            
            # 2.  put 50 % in storage
            self.storage_buffer = mixing_buffer[:mixing_buffer.shape[0]//2]
        
            # 3. put other 50 % in a dataloader
            dataloader = iter(DataLoader(mixing_buffer[mixing_buffer.shape[0]//2:], batch_size=batch_size, shuffle=True))
        
        return dataloader
    
    
    def next_batch(self):
        """
        Get the next batch from the current DataLoader. 
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)