"""
Training Configuration Module

Defines the configuration for RLVR training, including:
- Model to train (base OSS model)
- Training hyperparameters
- OpenPipe and W&B settings
"""

from typing import Literal
from pydantic import BaseModel, Field


class OpenPipeConfig(BaseModel):
    """OpenPipe-specific configuration."""
    api_key: str | None = Field(default=None, description="OpenPipe API key (or use OPENPIPE_API_KEY env var)")
    base_url: str | None = Field(default=None, description="OpenPipe API base URL")
    dataset_name: str = Field(default="rlvr-tool-training", description="Name for the training dataset")
    model_slug: str = Field(default="tool-selector-v1", description="Slug for the fine-tuned model")


class WandBConfig(BaseModel):
    """Weights & Biases configuration."""
    project: str = Field(default="rlvr-tool-training", description="W&B project name")
    entity: str | None = Field(default=None, description="W&B entity (team/user)")
    tags: list[str] = Field(default_factory=lambda: ["rlvr", "tool-calling"])
    log_model: bool = Field(default=True, description="Whether to log model artifacts")


class RewardHyperparameters(BaseModel):
    """
    Hyperparameters for RLVR training.
    
    These control how the model learns from the verifiable reward signals.
    """
    batch_size: int | Literal["auto"] = Field(default="auto", description="Training batch size")
    learning_rate_multiplier: float = Field(default=1.0, ge=0.1, le=10.0, description="Learning rate multiplier")
    num_epochs: int = Field(default=3, ge=1, le=10, description="Number of training epochs")


class TrainingConfig(BaseModel):
    """
    Main configuration for RLVR tool call training.
    
    This is the central configuration object that specifies:
    1. Which model to train (base_model)
    2. Training hyperparameters
    3. OpenPipe and W&B integration settings
    
    Example:
        config = TrainingConfig(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            hyperparameters=RewardHyperparameters(num_epochs=3),
        )
    """
    # === MODEL SPECIFICATION ===
    # This is the key input: which OSS model to fine-tune
    base_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="The base model to fine-tune. Supported models vary by provider. "
                    "Common choices: meta-llama/Llama-3.1-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3"
    )
    
    # === TRAINING HYPERPARAMETERS ===
    hyperparameters: RewardHyperparameters = Field(
        default_factory=RewardHyperparameters,
        description="Training hyperparameters for reward-based learning"
    )
    
    # === INTEGRATION SETTINGS ===
    openpipe: OpenPipeConfig = Field(
        default_factory=OpenPipeConfig,
        description="OpenPipe integration settings"
    )
    wandb: WandBConfig = Field(
        default_factory=WandBConfig,
        description="Weights & Biases settings for experiment tracking"
    )
    
    # === TRAINING OPTIONS ===
    train_test_split: float = Field(
        default=0.1, ge=0.0, le=0.5,
        description="Fraction of data to use for test/validation"
    )
    min_samples_required: int = Field(
        default=10,
        description="Minimum number of training samples required"
    )
    
    @classmethod
    def for_llama_8b(cls, **kwargs) -> "TrainingConfig":
        """Create config optimized for Llama 3.1 8B."""
        defaults = {
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "hyperparameters": RewardHyperparameters(
                batch_size="auto",
                learning_rate_multiplier=1.0,
                num_epochs=3
            )
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def for_mistral_7b(cls, **kwargs) -> "TrainingConfig":
        """Create config optimized for Mistral 7B."""
        defaults = {
            "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
            "hyperparameters": RewardHyperparameters(
                batch_size="auto",
                learning_rate_multiplier=0.8,
                num_epochs=3
            )
        }
        defaults.update(kwargs)
        return cls(**defaults)
