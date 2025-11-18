# core/atl_contract.py
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class AuditRecord:
    epoch: int
    ipfs_hash: str
    merkle_root: bytes
    client_ids: List[int]
    timestamp: float
    tx_hash: str

class ATLContract:
    """Audit Trail Layer - simulates Ethereum public blockchain."""
    
    def __init__(self, config):
        self.config = config
        self.audit_log: List[AuditRecord] = []
        self.committed_checkpoints: Dict[int, bytes] = {}
        self.transaction_counter = 0
        
    def commit_aggregation(
        self, 
        epoch: int, 
        ipfs_hash: str, 
        merkle_root: bytes,
        client_ids: List[int]
    ) -> str:
        """Commit aggregation record to audit trail."""
        # Simulate blockchain transaction
        self.transaction_counter += 1
        tx_hash = f"0x{hashlib.sha256(f'{epoch}{ipfs_hash}'.encode()).hexdigest()}"
        
        record = AuditRecord(
            epoch=epoch,
            ipfs_hash=ipfs_hash,
            merkle_root=merkle_root,
            client_ids=client_ids,
            timestamp=time.time(),
            tx_hash=tx_hash
        )
        
        self.audit_log.append(record)
        self.committed_checkpoints[epoch] = merkle_root
        
        print(f"🔗 ATL Commit: Epoch {epoch}, TX: {tx_hash[:16]}..., {len(client_ids)} clients")
        return tx_hash
    
    def verify_checkpoint(self, epoch: int, merkle_root: bytes) -> bool:
        """Verify if a checkpoint exists in audit trail."""
        stored = self.committed_checkpoints.get(epoch)
        if stored is None:
            return False
        return stored == merkle_root
    
    def get_audit_trail(self, start_epoch: int, end_epoch: int) -> List[AuditRecord]:
        """Retrieve audit trail for a range of epochs."""
        return [
            record for record in self.audit_log 
            if start_epoch <= record.epoch <= end_epoch
        ]