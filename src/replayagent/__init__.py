"""replayagent — record, replay, and regression-test AI agent trajectories."""

from replayagent.cases import (
    RegressionCase,
    check_case,
    list_cases,
    load_case,
    promote_run,
    promote_trajectory,
    save_case,
)
from replayagent.diff import TrajectoryDiff, diff_trajectories
from replayagent.ingest import (
    Failure,
    FailureCluster,
    IngestError,
    ProductionTrace,
    cluster_failures,
    cluster_traces,
    detect_failure,
    detect_failures,
    draft_case_for_cluster,
    draft_cases,
    list_drafts,
    load_draft,
    load_traces,
    parse_traces,
    promote_draft,
)
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
    "Failure",
    "FailureCluster",
    "IngestError",
    "ProductionTrace",
    "Judge",
    "JudgeResult",
    "LLMJudge",
    "Recorder",
    "RegressionCase",
    "SemanticSimilarityJudge",
    "Step",
    "StepType",
    "ToolCallJudge",
    "Trajectory",
    "TrajectoryDiff",
    "check_case",
    "cluster_failures",
    "cluster_traces",
    "detect_failure",
    "detect_failures",
    "diff_trajectories",
    "draft_case_for_cluster",
    "draft_cases",
    "list_cases",
    "list_drafts",
    "load_draft",
    "load_traces",
    "parse_traces",
    "promote_draft",
    "list_runs",
    "load_case",
    "load_run",
    "promote_run",
    "promote_trajectory",
    "record",
    "record_step",
    "replay",
    "save_case",
    "save_trajectory",
]

__version__ = "0.1.0"
