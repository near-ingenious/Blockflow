# sharding/shard_manager.py
from typing import List, Dict, Tuple
import hashlib

class ShardManager:
    """Manages hierarchical sharding architecture."""
    
    def __init__(self, num_shards: int, config):
        self.num_shards = num_shards
        self.config = config
        self.shard_assignments: Dict[int, int] = {}  # client_id -> shard_id
        self.shard_aggregators: Dict[int, List[int]] = {}  # shard_id -> [aggregator_ids]
        self.beacon_chain = []
        
    def assign_client_to_shard(self, client_id: int, geo_location: str, reputation: float) -> int:
        """Assign client to shard based on geography and reputation."""
        # Hash-based assignment with geo and reputation factors
        geo_hash = int(hashlib.sha256(geo_location.encode()).hexdigest(), 16)
        rep_score = int(reputation * 1000)
        
        shard_id = ((client_id * self.num_shards) // self.config.num_clients + geo_hash + rep_score) % self.num_shards
        
        self.shard_assignments[client_id] = shard_id
        
        if shard_id not in self.shard_aggregators:
            self.shard_aggregators[shard_id] = []
        
        return shard_id
    
    def get_shard_members(self, shard_id: int) -> List[int]:
        """Get all clients in a shard."""
        return [cid for cid, sid in self.shard_assignments.items() if sid == shard_id]
    
    def get_aggregator_for_shard(self, shard_id: int) -> int:
        """Get primary aggregator for shard."""
        if shard_id not in self.shard_aggregators:
            return shard_id  # Default: shard_id as aggregator_id
        aggregators = self.shard_aggregators[shard_id]
        return aggregators[0] if aggregators else shard_id
    
    def create_beacon_checkpoint(self, shard_states: Dict[int, bytes]) -> bytes:
        """Create beacon chain checkpoint from shard states."""
        # Combine shard Merkle roots
        sorted_shards = sorted(shard_states.items())
        combined = b"".join([root for _, root in sorted_shards])
        checkpoint = hashlib.sha256(combined).digest()
        
        self.beacon_chain.append({
            'epoch': len(self.beacon_chain),
            'checkpoint': checkpoint,
            'shard_states': shard_states
        })
        
        return checkpoint
    
    def rebalance_shards(self, client_reputations: Dict[int, float]):
        """Dynamically rebalance shards based on reputation."""
        # Simple rebalance: move high-reputation clients to balance load
        shard_loads = {i: len(self.get_shard_members(i)) for i in range(self.num_shards)}
        avg_load = sum(shard_loads.values()) / self.num_shards
        
        # Identify overloaded and underloaded shards
        overloaded = [sid for sid, load in shard_loads.items() if load > avg_load * 1.2]
        underloaded = [sid for sid, load in shard_loads.items() if load < avg_load * 0.8]
        
        if overloaded and underloaded:
            # Move highest reputation client from overloaded to underloaded
            for src_shard in overloaded:
                members = self.get_shard_members(src_shard)
                if members:
                    # Find client with highest reputation
                    best_client = max(members, key=lambda cid: client_reputations.get(cid, 0))
                    dest_shard = underloaded[0]
                    
                    self.shard_assignments[best_client] = dest_shard
                    print(f"🔄 Rebalanced client {best_client} from shard {src_shard} to {dest_shard}")
                    break