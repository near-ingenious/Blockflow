# core/reputation.py
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score
import random

class ReputationManager:
    def __init__(self, config):
        self.config = config
        self.contribution_history: Dict[int, List[float]] = {}
        self.shapley_cache: Dict[int, float] = {}
        
    def calculate_shapley_values(
        self, 
        client_updates: Dict[int, Dict], 
        validation_fn,
        global_model
    ) -> Dict[int, float]:
        """Calculate Shapley values using Monte Carlo approximation."""
        client_ids = list(client_updates.keys())
        n_clients = len(client_ids)
        
        if n_clients == 0:
            return {}
        
        shapley_values = {cid: 0.0 for cid in client_ids}
        
        # Monte Carlo sampling
        for _ in range(self.config.shapley_samples):
            # Random permutation
            permutation = random.sample(client_ids, n_clients)
            marginal_values = {}
            
            # Calculate marginal contribution for each client
            for i, client_id in enumerate(permutation):
                # Coalition without current client
                coalition_without = permutation[:i]
                # Coalition with current client
                coalition_with = permutation[:i+1]
                
                # Get utility scores
                if len(coalition_without) == 0:
                    utility_without = validation_fn(global_model)
                else:
                    model_without = self._simulate_aggregation(
                        {cid: client_updates[cid] for cid in coalition_without}
                    )
                    utility_without = validation_fn(model_without)
                
                model_with = self._simulate_aggregation(
                    {cid: client_updates[cid] for cid in coalition_with}
                )
                utility_with = validation_fn(model_with)
                
                marginal_values[client_id] = utility_with - utility_without
            
            # Add to Shapley values
            for client_id in client_ids:
                shapley_values[client_id] += marginal_values.get(client_id, 0)
        
        # Average over samples
        for client_id in client_ids:
            shapley_values[client_id] /= self.config.shapley_samples
        
        # Cache results
        self.shapley_cache.update(shapley_values)
        
        return shapley_values
    
    def _simulate_aggregation(self, updates: Dict[int, Dict]) -> Dict:
        """Simulate aggregation for utility calculation."""
        # Simplified aggregation for validation
        if not updates:
            return {}
        
        # Average the updates
        aggregated = {}
        for update in updates.values():
            for key, value in update.items():
                if key not in aggregated:
                    aggregated[key] = torch.zeros_like(value)
                aggregated[key] += value
        
        for key in aggregated:
            aggregated[key] /= len(updates)
        
        return aggregated
    
    def calculate_rewards(
        self, 
        shapley_values: Dict[int, float], 
        reputations: Dict[int, float]
    ) -> Dict[int, float]:
        """Calculate token rewards based on Shapley values and reputation."""
        rewards = {}
        total_weight = 0
        
        # Calculate weighted contributions
        weighted_contributions = {}
        for client_id in shapley_values:
            weight = shapley_values[client_id] * reputations.get(client_id, self.config.initial_reputation)
            weighted_contributions[client_id] = max(weight, 0)
            total_weight += weighted_contributions[client_id]
        
        # Allocate rewards proportionally
        for client_id in shapley_values:
            if total_weight > 0:
                reward = (weighted_contributions[client_id] / total_weight) * self.config.total_reward_per_round
            else:
                reward = self.config.total_reward_per_round / len(shapley_values)
            
            # Ensure individual rationality
            reward = max(reward, self.config.participation_cost)
            rewards[client_id] = reward
        
        return rewards