"""replayagent — record, replay, and regression-test AI agent trajectories."""

from replayagent.diff import TrajectoryDiff, diff_trajectories
from replayagent.judges import (
    ContainsJudge,
    ExactMatchJudge,
    Judge,
    JudgeResult,
    LLMJudge,
    SemanticSimilarityJudge,
    ToolCallJudge,
)
from replayagent.recorder import Recorder, list_runs, load_run, record, record_step, save_trajectory
from replayagent.replay import replay
from replayagent.trajectory import Step, StepType, Trajectory

__all__ = [
    "ContainsJudge",
    "ExactMatchJudge",
    "Judge",
    "JudgeResult",
    "LLMJudge",
    "Recorder",
    "SemanticSimilarityJudge",
    "Step",
    "StepType",
    "ToolCallJudge",
    "Trajectory",
    "TrajectoryDiff",
    "diff_trajectories",
    "list_runs",
    "load_run",
    "record",
    "record_step",
    "replay",
    "save_trajectory",
]

__version__ = "0.1.0"
