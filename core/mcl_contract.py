# core/mcl_contract.py
import time
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class ModelUpdate:
    client_id: int
    merkle_root: bytes
    zk_proof: bytes
    signature: bytes
    timestamp: float
    reputation: float = 0.5

class MCLContract:
    """Model Consensus Layer - simulates Hyperledger Fabric smart contract."""
    
    def __init__(self, config):
        self.config = config
        self.updates: Dict[int, ModelUpdate] = {}
        self.reputations: Dict[int, float] = {}
        self.merkle_roots: List[bytes] = []
        self.current_epoch = 0
        self.aggregation_ready = False
        self.lock = threading.Lock()
        self.validators = set()
        self.pending_aggregations = {}
        
    def register_client(self, client_id: int, public_key: bytes):
        """Register a new client with initial reputation."""
        with self.lock:
            if client_id not in self.reputations:
                self.reputations[client_id] = self.config.initial_reputation
                print(f"✅ Registered client {client_id} with reputation {self.config.initial_reputation}")
    
    def submit_update(
        self, 
        client_id: int, 
        merkle_root: bytes, 
        zk_proof: bytes,
        signature: bytes,
        public_key: bytes
    ) -> bool:
        """Submit model update for verification."""
        with self.lock:
            # Check reputation threshold
            if self.reputations.get(client_id, 0) < self.config.min_reputation:
                print(f"❌ Client {client_id} reputation too low: {self.reputations.get(client_id, 0)}")
                return False
            
            # Verify ZK proof (simulated)
            if self.config.use_zksnarks:
                if not self._verify_zk_proof(zk_proof, merkle_root):
                    self._slash_reputation(client_id)
                    print(f"❌ Invalid ZK proof for client {client_id}")
                    return False
            
            # Verify signature
            if not self._verify_signature(client_id, merkle_root + zk_proof, signature, public_key):
                self._slash_reputation(client_id)
                print(f"❌ Invalid signature for client {client_id}")
                return False
            
            # Store update
            update = ModelUpdate(
                client_id=client_id,
                merkle_root=merkle_root,
                zk_proof=zk_proof,
                signature=signature,
                timestamp=time.time(),
                reputation=self.reputations[client_id]
            )
            
            self.updates[client_id] = update
            self.merkle_roots.append(merkle_root)
            
            print(f"✅ Accepted update from client {client_id} (reputation: {self.reputations[client_id]:.2f})")
            return True
    
    def _verify_zk_proof(self, zk_proof: bytes, merkle_root: bytes) -> bool:
        """Simulated ZK proof verification."""
        # In real implementation: call zk-SNARK verification circuit
        return len(zk_proof) > 0
    
    def _verify_signature(self, client_id: int, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Simulated signature verification."""
        # In real implementation: use cryptographic library
        return hashlib.sha256(signature).hexdigest()[:8] != "deadbeef"
    
    def _slash_reputation(self, client_id: int):
        """Slash reputation for misbehavior."""
        current = self.reputations.get(client_id, self.config.initial_reputation)
        new_reputation = max(self.config.min_reputation, current * 0.9)
        self.reputations[client_id] = new_reputation
        print(f"🔨 Slashed client {client_id} reputation: {current:.2f} → {new_reputation:.2f}")
    
    def compute_aggregate_merkle_root(self) -> bytes:
        """Compute Merkle root of all client Merkle roots."""
        if not self.merkle_roots:
            return b""
        
        # Sort for deterministic ordering
        sorted_roots = sorted(self.merkle_roots)
        combined = b"".join(sorted_roots)
        return hashlib.sha256(combined).digest()
    
    def finalize_aggregation(self, ipfs_hash: str, merkle_proof: bytes, aggregator_id: int) -> bool:
        """Finalize aggregation and commit to ATL."""
        with self.lock:
            if not self.updates:
                return False
            
            agg_root = self.compute_aggregate_merkle_root()
            
            # Verify Merkle proof
            if not self._verify_merkle_proof(ipfs_hash, agg_root, merkle_proof):
                print(f"❌ Invalid Merkle proof from aggregator {aggregator_id}")
                return False
            
            # Store pending aggregation
            self.pending_aggregations[self.current_epoch] = {
                'ipfs_hash': ipfs_hash,
                'agg_root': agg_root,
                'client_ids': list(self.updates.keys()),
                'timestamp': time.time()
            }
            
            print(f"✅ Aggregation finalized for epoch {self.current_epoch} with {len(self.updates)} updates")
            self.aggregation_ready = True
            return True
    
    def _verify_merkle_proof(self, ipfs_hash: str, agg_root: bytes, proof: bytes) -> bool:
        """Simulated Merkle proof verification."""
        # In real implementation: verify against IPFS-stored model
        return len(proof) > 0
    
    def get_updates(self) -> Dict[int, ModelUpdate]:
        """Get all verified updates."""
        with self.lock:
            return self.updates.copy()
    
    def clear_epoch(self):
        """Clear updates for next epoch."""
        with self.lock:
            self.updates.clear()
            self.merkle_roots.clear()
            self.aggregation_ready = False
            self.current_epoch += 1
    
    def update_reputation(self, client_id: int, quality_score: float):
        """Update reputation based on contribution quality."""
        with self.lock:
            current = self.reputations.get(client_id, self.config.initial_reputation)
            if quality_score > 0.7:
                new_reputation = min(self.config.max_reputation, current + self.config.reputation_reward)
                print(f"⭐ Increased reputation for client {client_id}: {current:.2f} → {new_reputation:.2f}")
            elif quality_score < 0.3:
                new_reputation = max(self.config.min_reputation, current - 0.05)
                print(f"🔻 Decreased reputation for client {client_id}: {current:.2f} → {new_reputation:.2f}")
            else:
                new_reputation = current
            
            self.reputations[client_id] = new_reputation