import os
import sys

# Add project root to Python path (MUST be before any other imports)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now all imports work as absolute
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
import time
import matplotlib.pyplot as plt
import random

from core.client import FLClient
from core.aggregator import Aggregator
from core.mcl_contract import MCLContract
from core.atl_contract import ATLContract
from core.reputation import ReputationManager
from sharding.shard_manager import ShardManager
from crypto.zksnark import ZKProver, ZKVerifier


class BlockFlowSimulator:
    def __init__(self, config: BlockFlowConfig):
        self.config = config
        self.clients: Dict[int, FLClient] = {}
        self.aggregators: Dict[int, Aggregator] = {}
        self.mcl = MCLContract(config)
        self.atl = ATLContract(config)
        self.reputation_mgr = ReputationManager(config)
        self.shard_manager = ShardManager(config.num_shards, config)

        # Create global model
        self.global_model = self._create_model()
        self.global_weights = self.global_model.state_dict()

        # Metrics
        self.metrics = {
            "accuracy": [],
            "participation_rate": [],
            "reputation_scores": [],
            "rewards": [],
            "throughput": [],
            "latency": [],
        }

    def _create_model(self) -> nn.Module:
        """Create neural network model."""
        if self.config.model_architecture == "cnn":
            return nn.Sequential(
                nn.Conv2d(1, 32, 3, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(9216, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )
        else:
            return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    def setup_clients(self, num_clients: int, data_loaders):
        """Initialize FL clients."""
        for i in range(num_clients):
            client = FLClient(
                client_id=i,
                data_loader=data_loaders[i] if i < len(data_loaders) else None,
                model=self._create_model(),
                config=self.config,
            )
            self.clients[i] = client

            # Register with MCL
            self.mcl.register_client(i, client.public_key)

            # Assign to shard
            geo = f"region_{i % 5}"
            self.shard_manager.assign_client_to_shard(i, geo, client.reputation)

        print(
            f"✅ Setup {len(self.clients)} clients across {self.config.num_shards} shards"
        )

    def setup_aggregators(self):
        """Initialize aggregators."""
        for i in range(self.config.num_shards):
            aggregator = Aggregator(i, self.config)
            self.aggregators[i] = aggregator
            self.shard_manager.shard_aggregators[i] = [i]

        print(f"✅ Setup {len(self.aggregators)} aggregators")

    def run_training_round(self, round_num: int) -> Dict[str, Any]:
        """Execute one federated learning round."""
        print(f"\n▶️  Starting training round {round_num}")
        start_time = time.time()

        # Phase 1: Local training and commitment
        client_updates = {}
        commitments = {}

        for client_id, client in tqdm(self.clients.items(), desc="Client updates"):
            # Simulate participation based on reputation
            if random.random() > client.reputation:
                continue

            # Local training
            updates = client.local_training(self.global_weights)

            # Generate cryptographic commitments
            merkle_root, zk_proof, signature = client.generate_commitment(updates)

            # Submit to MCL
            if self.mcl.submit_update(
                client_id, merkle_root, zk_proof, signature, client.public_key
            ):
                client_updates[client_id] = updates
                commitments[client_id] = (merkle_root, zk_proof, signature)

        participation_rate = len(client_updates) / len(self.clients)
        print(
            f"📊 Participation rate: {participation_rate:.2%} ({len(client_updates)}/{len(self.clients)})"
        )

        # Phase 2: Shard-level aggregation
        shard_updates = {}
        shard_reputations = {}
        shard_sizes = {}

        for shard_id in range(self.config.num_shards):
            shard_clients = self.shard_manager.get_shard_members(shard_id)
            shard_client_updates = {
                cid: client_updates[cid]
                for cid in shard_clients
                if cid in client_updates
            }

            if not shard_client_updates:
                continue

            aggregator = self.aggregators[shard_id]

            # Get reputations and dataset sizes
            reputations = [self.clients[cid].reputation for cid in shard_client_updates]
            sizes = [len(self.clients[cid].data_loader) for cid in shard_client_updates]

            # Aggregate
            aggregated_update, ipfs_hash = aggregator.aggregate_models(
                list(shard_client_updates.values()), reputations, sizes
            )

            shard_updates[shard_id] = aggregated_update
            shard_reputations[shard_id] = np.mean(reputations)
            shard_sizes[shard_id] = sum(sizes)

        # Phase 3: Global aggregation
        if shard_updates:
            global_aggregator = Aggregator(-1, self.config)  # Global aggregator
            final_update, final_ipfs_hash = global_aggregator.aggregate_models(
                list(shard_updates.values()),
                list(shard_reputations.values()),
                list(shard_sizes.values()),
            )

            # Update global model
            for name, param in final_update.items():
                self.global_weights[name] += param

            # Create Merkle proof
            agg_root = self.mcl.compute_aggregate_merkle_root()

            # Finalize on MCL
            if self.mcl.finalize_aggregation(final_ipfs_hash, b"merkle_proof_sim", -1):
                # Commit to ATL
                tx_hash = self.atl.commit_aggregation(
                    self.mcl.current_epoch,
                    final_ipfs_hash,
                    agg_root,
                    list(client_updates.keys()),
                )

                # Phase 4: Reward distribution
                self._distribute_rewards(client_updates)

                # Phase 5: Reputation update
                self._update_reputations(client_updates)

        # Clear for next epoch
        self.mcl.clear_epoch()

        # Record metrics
        round_time = time.time() - start_time
        throughput = len(client_updates) / round_time

        self.metrics["participation_rate"].append(participation_rate)
        self.metrics["throughput"].append(throughput)
        self.metrics["latency"].append(round_time)

        return {
            "round": round_num,
            "participation_rate": participation_rate,
            "throughput": throughput,
            "latency": round_time,
            "active_clients": len(client_updates),
        }

    def _distribute_rewards(self, client_updates: Dict[int, Dict]):
        """Calculate and distribute rewards."""

        # Calculate Shapley values (simplified)
        def validation_fn(model):
            # Simulate validation accuracy
            return np.random.beta(2, 2)  # Random quality score

        shapley_values = self.reputation_mgr.calculate_shapley_values(
            client_updates, validation_fn, self.global_model
        )

        # Calculate rewards
        reputations = {cid: self.clients[cid].reputation for cid in client_updates}
        rewards = self.reputation_mgr.calculate_rewards(shapley_values, reputations)

        # Distribute (simulated)
        for client_id, reward in rewards.items():
            self.metrics["rewards"].append(reward)
            print(f"💰 Client {client_id} reward: {reward:.2f} tokens")

    def _update_reputations(self, client_updates: Dict[int, Dict]):
        """Update client reputations based on contribution quality."""
        for client_id in client_updates:
            # Simulate quality score
            quality_score = np.random.beta(2, 2)
            self.mcl.update_reputation(client_id, quality_score)
            self.clients[client_id].reputation = self.mcl.reputations[client_id]

    def run_simulation(self, num_rounds: int):
        """Run complete FL simulation."""
        print(f"🚀 Starting BlockFlow simulation for {num_rounds} rounds")
        print(f"   Clients: {self.config.num_clients}")
        print(f"   Shards: {self.config.num_shards}")
        print(f"   ZK-SNARKs: {self.config.use_zksnarks}")

        results = []

        for round_num in range(num_rounds):
            result = self.run_training_round(round_num)
            results.append(result)

            # Rebalance shards periodically
            if round_num > 0 and round_num % 10 == 0:
                self.shard_manager.rebalance_shards(
                    {cid: c.reputation for cid, c in self.clients.items()}
                )

        self._plot_metrics()
        return results

    def _plot_metrics(self):
        """Plot simulation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Participation rate
        axes[0, 0].plot(self.metrics["participation_rate"])
        axes[0, 0].set_title("Client Participation Rate")
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Participation Rate")

        # Throughput
        axes[0, 1].plot(self.metrics["throughput"])
        axes[0, 1].set_title("System Throughput")
        axes[0, 1].set_xlabel("Round")
        axes[0, 1].set_ylabel("Updates/sec")

        # Latency
        axes[1, 0].plot(self.metrics["latency"])
        axes[1, 0].set_title("Round Latency")
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("Latency (s)")

        # Reputation distribution
        if self.clients:
            reputations = [c.reputation for c in self.clients.values()]
            axes[1, 1].hist(reputations, bins=20)
            axes[1, 1].set_title("Reputation Distribution")
            axes[1, 1].set_xlabel("Reputation Score")
            axes[1, 1].set_ylabel("Number of Clients")

        plt.tight_layout()
        plt.savefig("blockflow_metrics.png")
        print("📊 Metrics saved to blockflow_metrics.png")
        plt.close()


# Usage Example
if __name__ == "__main__":
    # Configuration
    config = BlockFlowConfig(
        num_clients=100, num_shards=5, use_zksnarks=True, total_reward_per_round=1000.0
    )

    # Create simulator
    simulator = BlockFlowSimulator(config)

    # Setup mock data loaders
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    # Generate synthetic data
    X = torch.randn(1000, 1, 28, 28)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)

    data_loaders = []
    for i in range(config.num_clients):
        client_data = torch.utils.data.Subset(dataset, range(i * 10, (i + 1) * 10))
        data_loaders.append(DataLoader(client_data, batch_size=config.batch_size))

    simulator.setup_clients(config.num_clients, data_loaders)
    simulator.setup_aggregators()

    # Run simulation
    results = simulator.run_simulation(num_rounds=20)

    print("\n✅ Simulation complete!")
    print(f"Final participation rate: {results[-1]['participation_rate']:.2%}")
    print(
        f"Average throughput: {np.mean([r['throughput'] for r in results]):.2f} updates/sec"
    )
