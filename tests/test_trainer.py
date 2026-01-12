"""Tests for the RLVR trainer."""

import pytest
from rlvr_tool_training.config import TrainingConfig, ToolRegistry, ToolDefinition
from rlvr_tool_training.core import (
    RLVRTrainer,
    Trajectory,
    ToolCall,
    VerifiableReward,
    TrajectoryGenerator,
)


@pytest.fixture
def sample_config():
    return TrainingConfig(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        hyperparameters={
            "num_epochs": 3,
            "batch_size": "auto"
        },
        openpipe={
            "dataset_name": "test-dataset",
            "model_slug": "test-model"
        },
        wandb={
            "project": "test-project"
        }
    )


@pytest.fixture
def sample_tools():
    return ToolRegistry(tools=[
        ToolDefinition(name="read_file", description="Read a file"),
        ToolDefinition(name="write_file", description="Write a file"),
    ])


@pytest.fixture
def sample_trajectories():
    generator = TrajectoryGenerator()
    return [
        generator.create_trajectory(
            user_message=f"Test message {i}",
            chosen_tool="read_file",
            chosen_args={"path": f"file{i}.txt"},
            reward=VerifiableReward.perfect()
        )
        for i in range(15)  # Enough to meet minimum
    ]


class TestTrainingConfig:
    def test_default_config(self):
        config = TrainingConfig()
        
        assert config.base_model == "meta-llama/Llama-3.1-8B-Instruct"
        assert config.hyperparameters.num_epochs == 3
    
    def test_for_llama_8b(self):
        config = TrainingConfig.for_llama_8b()
        
        assert "llama" in config.base_model.lower()
    
    def test_for_mistral_7b(self):
        config = TrainingConfig.for_mistral_7b()
        
        assert "mistral" in config.base_model.lower()


class TestRLVRTrainer:
    def test_initialization(self, sample_config, sample_tools):
        trainer = RLVRTrainer(sample_config, sample_tools, dry_run=True)
        
        assert trainer.config == sample_config
        assert trainer.tool_registry == sample_tools
        assert trainer.dry_run is True
    
    def test_add_trajectories(self, sample_config, sample_tools, sample_trajectories):
        trainer = RLVRTrainer(sample_config, sample_tools, dry_run=True)
        
        trainer.add_trajectories(sample_trajectories)
        
        assert len(trainer.trajectories) == len(sample_trajectories)
    
    def test_add_single_trajectory(self, sample_config, sample_tools):
        trainer = RLVRTrainer(sample_config, sample_tools, dry_run=True)
        trajectory = Trajectory(
            user_message="Test",
            tool_call=ToolCall(name="read_file", arguments={}),
            reward=VerifiableReward.perfect()
        )
        
        trainer.add_trajectory(trajectory)
        
        assert len(trainer.trajectories) == 1
    
    def test_get_training_summary(self, sample_config, sample_tools, sample_trajectories):
        trainer = RLVRTrainer(sample_config, sample_tools, dry_run=True)
        trainer.add_trajectories(sample_trajectories)
        
        summary = trainer.get_training_summary()
        
        assert summary["model"] == sample_config.base_model
        assert summary["num_trajectories"] == len(sample_trajectories)
        assert "reward_stats" in summary
    
    def test_train_dry_run(self, sample_config, sample_tools, sample_trajectories):
        trainer = RLVRTrainer(sample_config, sample_tools, dry_run=True)
        trainer.add_trajectories(sample_trajectories)
        
        results = trainer.train()
        
        assert "dataset_id" in results
        assert "model" in results
        assert results["model"]["status"] == "dry_run"
    
    def test_train_insufficient_samples(self, sample_config, sample_tools):
        config = TrainingConfig(min_samples_required=10)
        trainer = RLVRTrainer(config, sample_tools, dry_run=True)
        
        # Only add 5 trajectories (less than minimum)
        generator = TrajectoryGenerator()
        for i in range(5):
            trainer.add_trajectory(generator.create_trajectory(
                user_message=f"Test {i}",
                chosen_tool="read_file",
                chosen_args={},
                reward=VerifiableReward.perfect()
            ))
        
        with pytest.raises(ValueError) as exc_info:
            trainer.train()
        
        assert "at least" in str(exc_info.value).lower()
    
    def test_train_results_structure(self, sample_config, sample_tools, sample_trajectories):
        trainer = RLVRTrainer(sample_config, sample_tools, dry_run=True)
        trainer.add_trajectories(sample_trajectories)
        
        results = trainer.train()
        
        # Check expected result structure
        assert "config" in results
        assert "steps" in results
        assert "dataset_id" in results
        assert "model" in results
        
        # Check steps were executed
        step_names = [s["name"] for s in results["steps"]]
        assert "init_wandb" in step_names
        assert "create_dataset" in step_names
        assert "upload_trajectories" in step_names
        assert "start_training" in step_names
