"""The training stack: budget config, the host-agnostic loop, checkpoints.

`python -m deluge.train` runs it; trainer.py holds the torch adapter and CLI and
is the only module here that touches torch.
"""

from .checkpoint import Checkpointer
from .config import TrainConfig, load_train_config
from .loop import Outcome, ResumeMismatch, TrainLoop, jsonl_logger

__all__ = ["Checkpointer", "TrainConfig", "load_train_config",
           "Outcome", "ResumeMismatch", "TrainLoop", "jsonl_logger"]
