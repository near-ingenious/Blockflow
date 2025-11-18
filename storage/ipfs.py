# storage/ipfs.py
import pickle
import hashlib
from typing import Any, Optional, Dict
import json
import os

class IPFSStorage:
    """Simulated IPFS storage layer."""
    
    def __init__(self, storage_dir: str = "./ipfs_storage"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.cache: Dict[str, Any] = {}
    
    def store_model(self, model: Dict, pin: bool = True) -> str:
        """Store model and return IPFS hash."""
        # Serialize model
        model_bytes = pickle.dumps(model)
        
        # Generate content hash (CID in real IPFS)
        content_hash = hashlib.sha256(model_bytes).hexdigest()
        ipfs_hash = f"Qm{content_hash[:44]}"  # Simulate IPFS hash format
        
        # Store on disk
        filepath = os.path.join(self.storage_dir, ipfs_hash)
        with open(filepath, 'wb') as f:
            f.write(model_bytes)
        
        # Cache
        self.cache[ipfs_hash] = model
        
        print(f"💾 Stored model on IPFS: {ipfs_hash[:16]}...")
        return ipfs_hash
    
    def retrieve_model(self, ipfs_hash: str) -> Optional[Dict]:
        """Retrieve model from IPFS."""
        # Check cache
        if ipfs_hash in self.cache:
            return self.cache[ipfs_hash]
        
        # Load from disk
        filepath = os.path.join(self.storage_dir, ipfs_hash)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
                self.cache[ipfs_hash] = model
                return model
        
        print(f"❌ Model not found: {ipfs_hash}")
        return None
    
    def store_audit_log(self, log_data: Dict) -> str:
        """Store audit log entry."""
        log_bytes = json.dumps(log_data, sort_keys=True).encode()
        log_hash = hashlib.sha256(log_bytes).hexdigest()
        ipfs_hash = f"Qm{log_hash[:44]}"
        
        filepath = os.path.join(self.storage_dir, f"audit_{ipfs_hash}")
        with open(filepath, 'wb') as f:
            f.write(log_bytes)
        
        return ipfs_hash