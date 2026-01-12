"""
RLVR Trainer Module

Integrates OpenPipe for model fine-tuning and Weights & Biases for experiment tracking.
This is the main orchestrator that ties together:
- Training configuration
- Tool definitions
- Trajectories with verifiable rewards
"""

import os
from typing import Any
import json

from ..config import TrainingConfig, ToolRegistry
from .trajectories import Trajectory


class RLVRTrainer:
    """
    RLVR Trainer that uses OpenPipe for fine-tuning and W&B for tracking.
    
    This class demonstrates how to wire together:
    1. A specified OSS model (via config.base_model)
    2. Available tools (via ToolRegistry)
    3. VR signals (via Trajectories with VerifiableRewards)
    
    Example:
        config = TrainingConfig(base_model="meta-llama/Llama-3.1-8B-Instruct")
        tools = create_coding_assistant_tools()
        
        trainer = RLVRTrainer(config, tools)
        trainer.add_trajectories([...])
        trainer.train()
    """
    
    def __init__(
        self, 
        config: TrainingConfig, 
        tool_registry: ToolRegistry,
        dry_run: bool = False
    ):
        """
        Initialize the RLVR trainer.
        
        Args:
            config: Training configuration specifying model, hyperparameters, etc.
            tool_registry: Registry of available tools
            dry_run: If True, don't actually call APIs (for testing)
        """
        self.config = config
        self.tool_registry = tool_registry
        self.dry_run = dry_run
        self.trajectories: list[Trajectory] = []
        
        # Initialize clients (lazy - only when needed)
        self._openpipe_client = None
        self._wandb_run = None
        self._dataset_id = None
    
    @property
    def openpipe_client(self):
        """Lazy initialization of OpenPipe client."""
        if self._openpipe_client is None and not self.dry_run:
            from openpipe import OpenPipe
            self._openpipe_client = OpenPipe(
                api_key=self.config.openpipe.api_key or os.environ.get("OPENPIPE_API_KEY"),
                base_url=self.config.openpipe.base_url
            )
        return self._openpipe_client
    
    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a single training trajectory."""
        self.trajectories.append(trajectory)
    
    def add_trajectories(self, trajectories: list[Trajectory]) -> None:
        """Add multiple training trajectories."""
        self.trajectories.extend(trajectories)
    
    def _init_wandb(self) -> Any:
        """Initialize Weights & Biases run for experiment tracking."""
        if self._wandb_run is not None:
            return self._wandb_run
        
        if self.dry_run:
            return None
        
        import wandb
        
        # Log configuration as W&B config
        wandb_config = {
            "base_model": self.config.base_model,
            "num_trajectories": len(self.trajectories),
            "num_tools": len(self.tool_registry.tools),
            "tool_names": [t.name for t in self.tool_registry.tools],
            "hyperparameters": self.config.hyperparameters.model_dump(),
            "train_test_split": self.config.train_test_split,
        }
        
        self._wandb_run = wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            tags=self.config.wandb.tags,
            config=wandb_config,
            job_type="training"
        )
        
        return self._wandb_run
    
    def _create_dataset(self) -> str:
        """Create OpenPipe dataset and return dataset ID."""
        if self.dry_run:
            return "dry-run-dataset-id"
        
        response = self.openpipe_client.create_dataset(
            name=self.config.openpipe.dataset_name
        )
        return response.dataset_id
    
    def _upload_trajectories(self, dataset_id: str) -> dict[str, Any]:
        """Upload trajectories to OpenPipe dataset."""
        tools = self.tool_registry.to_openai_tools()
        
        # Convert trajectories to OpenPipe format
        entries = [t.to_openpipe_entry(tools) for t in self.trajectories]
        
        if self.dry_run:
            return {
                "dataset_id": dataset_id,
                "num_entries": len(entries),
                "entries_preview": entries[:2] if entries else []
            }
        
        # Upload in batches (OpenPipe may have limits)
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            self.openpipe_client.create_dataset_entries(
                dataset_id=dataset_id,
                entries=batch
            )
            total_uploaded += len(batch)
        
        return {
            "dataset_id": dataset_id,
            "num_entries": total_uploaded
        }
    
    def _start_training(self, dataset_id: str) -> dict[str, Any]:
        """Start the training job on OpenPipe."""
        if self.dry_run:
            return {
                "model_id": "dry-run-model-id",
                "status": "dry_run",
                "config": {
                    "provider": "openpipeReward",
                    "base_model": self.config.base_model,
                    "hyperparameters": self.config.hyperparameters.model_dump()
                }
            }
        
        # Build training config for OpenPipe reward-based training
        hyperparams = {}
        if self.config.hyperparameters.batch_size != "auto":
            hyperparams["batch_size"] = self.config.hyperparameters.batch_size
        if self.config.hyperparameters.learning_rate_multiplier != 1.0:
            hyperparams["learning_rate_multiplier"] = self.config.hyperparameters.learning_rate_multiplier
        if self.config.hyperparameters.num_epochs != 3:
            hyperparams["num_epochs"] = self.config.hyperparameters.num_epochs
        
        training_config = {
            "provider": "openpipeReward",
            "base_model": self.config.base_model,
        }
        if hyperparams:
            training_config["hyperparameters"] = hyperparams
        
        response = self.openpipe_client.create_model(
            dataset_id=dataset_id,
            slug=self.config.openpipe.model_slug,
            training_config=training_config
        )
        
        return {
            "model_id": response.id if hasattr(response, 'id') else str(response),
            "status": "started",
            "config": training_config
        }
    
    def train(self) -> dict[str, Any]:
        """
        Execute the full RLVR training pipeline.
        
        Steps:
        1. Validate configuration and trajectories
        2. Initialize W&B tracking
        3. Create OpenPipe dataset
        4. Upload trajectories with VR signals
        5. Start training job
        6. Log results to W&B
        
        Returns:
            Dictionary with training job details
        """
        # Validate
        if len(self.trajectories) < self.config.min_samples_required:
            raise ValueError(
                f"Need at least {self.config.min_samples_required} trajectories, "
                f"got {len(self.trajectories)}"
            )
        
        results = {
            "config": {
                "base_model": self.config.base_model,
                "num_trajectories": len(self.trajectories),
                "num_tools": len(self.tool_registry.tools),
            },
            "steps": []
        }
        
        # Initialize W&B
        wandb_run = self._init_wandb()
        results["steps"].append({"name": "init_wandb", "status": "success"})
        
        # Create dataset
        dataset_id = self._create_dataset()
        self._dataset_id = dataset_id
        results["dataset_id"] = dataset_id
        results["steps"].append({"name": "create_dataset", "status": "success", "dataset_id": dataset_id})
        
        # Upload trajectories
        upload_result = self._upload_trajectories(dataset_id)
        results["steps"].append({"name": "upload_trajectories", "status": "success", **upload_result})
        
        # Log trajectory stats to W&B
        if wandb_run and not self.dry_run:
            import wandb
            
            # Log reward distribution
            rewards = [t.reward.score for t in self.trajectories]
            wandb.log({
                "reward_mean": sum(rewards) / len(rewards),
                "reward_min": min(rewards),
                "reward_max": max(rewards),
                "num_perfect_rewards": sum(1 for r in rewards if r == 1.0),
                "num_zero_rewards": sum(1 for r in rewards if r == 0.0),
            })
            
            # Log tool usage distribution
            tool_counts = {}
            for t in self.trajectories:
                tool_counts[t.tool_call.name] = tool_counts.get(t.tool_call.name, 0) + 1
            
            wandb.log({"tool_distribution": tool_counts})
        
        # Start training
        training_result = self._start_training(dataset_id)
        results["model"] = training_result
        results["steps"].append({"name": "start_training", "status": "success", **training_result})
        
        # Final W&B logging
        if wandb_run and not self.dry_run:
            import wandb
            wandb.log({
                "training_started": True,
                "model_id": training_result.get("model_id"),
            })
            
            # Log the training config as an artifact
            if self.config.wandb.log_model:
                artifact = wandb.Artifact(
                    name=f"training-config-{self.config.openpipe.model_slug}",
                    type="training-config"
                )
                with artifact.new_file("config.json") as f:
                    f.write(json.dumps(self.config.model_dump(), indent=2, default=str))
                wandb.log_artifact(artifact)
        
        return results
    
    def get_training_summary(self) -> dict[str, Any]:
        """Get a summary of the training configuration and data."""
        reward_scores = [t.reward.score for t in self.trajectories]
        
        return {
            "model": self.config.base_model,
            "tools": [t.name for t in self.tool_registry.tools],
            "num_trajectories": len(self.trajectories),
            "reward_stats": {
                "mean": sum(reward_scores) / len(reward_scores) if reward_scores else 0,
                "min": min(reward_scores) if reward_scores else 0,
                "max": max(reward_scores) if reward_scores else 0,
            },
            "hyperparameters": self.config.hyperparameters.model_dump(),
        }
