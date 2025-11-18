# core/client.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from ..crypto.merkle import MerkleTree
from ..crypto.zksnark import ZKProver, ZKVerifier

class FLClient:
    def __init__(self, client_id: int, data_loader, model: nn.Module, config, private_key: Optional[bytes] = None):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model = model
        self.config = config
        self.private_key = private_key or self._generate_key()
        self.public_key = self._derive_public_key()
        self.reputation = config.initial_reputation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def _generate_key(self) -> bytes:
        from cryptography.hazmat.primitives.asymmetric import rsa
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        return key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
    
    def _derive_public_key(self) -> bytes:
        # Simplified key derivation
        return hashlib.sha256(self.private_key).digest()
    
    def local_training(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform local training and return model update."""
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Calculate update (delta)
        updates = {}
        for name, param in self.model.named_parameters():
            updates[name] = param.data.cpu() - global_weights[name].cpu()
        
        return updates
    
    def generate_commitment(self, updates: Dict[str, torch.Tensor]) -> Tuple[bytes, bytes, bytes]:
        """Generate Merkle root, ZK proof, and signature."""
        # Flatten updates for Merkle tree
        update_vector = torch.cat([u.flatten() for u in updates.values()])
        update_bytes = update_vector.numpy().tobytes()
        
        # Merkle commitment
        merkle_tree = MerkleTree(update_bytes)
        merkle_root = merkle_tree.get_root()
        
        # ZK proof (simulated)
        if self.config.use_zksnarks:
            zk_prover = ZKProver()
            zk_proof = zk_prover.generate_proof(updates, merkle_root)
        else:
            zk_proof = b"zk_proof_placeholder"
        
        # Digital signature
        signature = self._sign(merkle_root + zk_proof)
        
        return merkle_root, zk_proof, signature
    
    def _sign(self, data: bytes) -> bytes:
        """Sign data with private key."""
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
        from cryptography.hazmat.backends import default_backend
        
        private_key = serialization.load_pem_private_key(
            self.private_key, password=None, backend=default_backend()
        )
        signature = private_key.sign(
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return signature
    
    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify signature using public key."""
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            
            public_key = serialization.load_pem_public_key(self.public_key)
            public_key.verify(
                signature,
                data,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            return True
        except:
            return False