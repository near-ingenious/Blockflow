# config.py
from pydantic import BaseModel
from typing import List, Dict, Any

class BlockFlowConfig(BaseModel):
    # Training parameters
    num_clients: int = 100
    num_shards: int = 5
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    
    # Model parameters
    model_architecture: str = "cnn"
    model_params: Dict[str, Any] = {"hidden_dim": 128, "num_classes": 10}
    
    # Blockchain parameters
    mcl_block_time: float = 0.2  # 200ms
    atl_block_time: float = 12.0  # 12s (Ethereum-like)
    confirmation_delay: int = 3
    
    # Reputation system
    initial_reputation: float = 0.5
    reputation_penalty: float = 0.1
    reputation_reward: float = 0.01
    min_reputation: float = 0.1
    max_reputation: float = 1.0
    
    # Incentive mechanism
    total_reward_per_round: float = 1000.0
    shapley_samples: int = 50
    participation_cost: float = 10.0
    
    # Cryptographic parameters
    use_zksnarks: bool = True
    use_merkle: bool = True
    
    # Performance tuning
    max_updates_per_second: int = 1200
    ipfs_timeout: int = 30