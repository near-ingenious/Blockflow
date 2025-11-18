# crypto/merkle.py
import hashlib
from typing import List, Tuple, Optional

class MerkleNode:
    def __init__(self, hash_val: bytes, left=None, right=None, data=None):
        self.hash = hash_val
        self.left = left
        self.right = right
        self.data = data

class MerkleTree:
    def __init__(self, data: bytes, chunk_size: int = 1024):
        self.chunk_size = chunk_size
        self.leaves = self._create_leaves(data)
        self.root = self._build_tree(self.leaves)
    
    def _create_leaves(self, data: bytes) -> List[MerkleNode]:
        """Create leaf nodes from data chunks."""
        leaves = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            leaf_hash = hashlib.sha256(b"\x00" + chunk).digest()
            leaves.append(MerkleNode(leaf_hash, data=chunk))
        return leaves
    
    def _build_tree(self, nodes: List[MerkleNode]) -> Optional[MerkleNode]:
        """Build Merkle tree from leaf nodes."""
        if not nodes:
            return None
        
        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else nodes[i]
                combined_hash = hashlib.sha256(b"\x01" + left.hash + right.hash).digest()
                parent = MerkleNode(combined_hash, left, right)
                next_level.append(parent)
            nodes = next_level
        
        return nodes[0] if nodes else None
    
    def get_root(self) -> bytes:
        """Get Merkle root hash."""
        return self.root.hash if self.root else b""
    
    def get_proof(self, chunk_index: int) -> List[bytes]:
        """Generate Merkle proof for a specific chunk."""
        proof = []
        nodes = self.leaves
        
        while len(nodes) > 1:
            sibling_index = chunk_index ^ 1  # XOR to find sibling
            if sibling_index < len(nodes):
                proof.append(nodes[sibling_index].hash)
            
            chunk_index //= 2
            next_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else nodes[i]
                next_level.append(MerkleNode(b"internal", left, right))
            nodes = next_level
        
        return proof
    
    def verify_proof(self, proof: List[bytes], root: bytes, chunk: bytes, chunk_index: int) -> bool:
        """Verify Merkle proof for a chunk."""
        current_hash = hashlib.sha256(b"\x00" + chunk).digest()
        
        for sibling_hash in proof:
            if chunk_index % 2 == 0:
                current_hash = hashlib.sha256(b"\x01" + current_hash + sibling_hash).digest()
            else:
                current_hash = hashlib.sha256(b"\x01" + sibling_hash + current_hash).digest()
            chunk_index //= 2
        
        return current_hash == root

    def verify_proof_by_hash(self, proof: List[bytes], root: bytes, chunk_hash: bytes, chunk_index: int) -> bool:
        """Verify proof using chunk hash directly."""
        current_hash = chunk_hash
        
        for sibling_hash in proof:
            if chunk_index % 2 == 0:
                current_hash = hashlib.sha256(b"\x01" + current_hash + sibling_hash).digest()
            else:
                current_hash = hashlib.sha256(b"\x01" + sibling_hash + current_hash).digest()
            chunk_index //= 2
        
        return current_hash == root