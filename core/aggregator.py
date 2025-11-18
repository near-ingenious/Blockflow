# core/aggregator.py
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np
from ..storage.ipfs import IPFSStorage

class Aggregator:
    def __init__(self, aggregator_id: int, config):
        self.aggregator_id = aggregator_id
        self.config = config
        self.storage = IPFSStorage()
        self.shard_members = []
        
    def aggregate_models(
        self, 
        updates: List[Dict[str, torch.Tensor]], 
        reputations: List[float],
        dataset_sizes: List[int]
    ) -> Tuple[nn.Module, str]:
        """Perform reputation-weighted FedAvg aggregation."""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Calculate weights based on reputation and dataset size
        total_weight = sum(rep * size for rep, size in zip(reputations, dataset_sizes))
        weights = [rep * size / total_weight for rep, size in zip(reputations, dataset_sizes)]
        
        # Initialize aggregated model
        aggregated_update = {name: torch.zeros_like(param) for name, param in updates[0].items()}
        
        # Weighted aggregation
        for update, weight in zip(updates, weights):
            for name, param in update.items():
                aggregated_update[name] += weight * param
        
        # Store on IPFS
        model_hash = self.storage.store_model(aggregated_update)
        
        return aggregated_update, model_hash
    
    def verify_merkle_proof(
        self, 
        ipfs_hash: str, 
        merkle_root: bytes, 
        merkle_proof: bytes
    ) -> bool:
        """Verify that IPFS-stored model matches Merkle root."""
        stored_model = self.storage.retrieve_model(ipfs_hash)
        model_bytes = self._model_to_bytes(stored_model)
        tree = MerkleTree(model_bytes)
        
        return tree.verify_proof(merkle_proof, merkle_root)
    
    def _model_to_bytes(self, model: Dict[str, torch.Tensor]) -> bytes:
        """Convert model to bytes for verification."""
        vector = torch.cat([v.flatten() for v in model.values()])
        return vector.numpy().tobytes()