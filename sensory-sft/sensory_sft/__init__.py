"""Sensory predictive-feedback trace collection (work-trial data engine).

Public surface:
    Registry        - annotate sense records from falmart's schema-registry.json
    SenseClient     - cursor-diff reader over /api/sense/log
    classify_delta  - reduce a record delta to one labeled Outcome
    Outcome         - the 4(+1) canonical click-outcome classes
    run_episode     - guarded rollout that emits (context, action) -> outcome
    RolloutConfig   - max_steps / sensory_on / loop-break knobs
    JsonlWriter     - append-only example sink
"""

from .registry import Registry, RpcAnnotation, RouteAnnotation
from .rollout import (
    Driver,
    EpisodeResult,
    JsonlWriter,
    Observation,
    Policy,
    RolloutConfig,
    TrainingExample,
    run_episode,
)
from .sense import Outcome, SenseClient, SenseDelta, classify_delta

__all__ = [
    "Registry",
    "RpcAnnotation",
    "RouteAnnotation",
    "SenseClient",
    "SenseDelta",
    "classify_delta",
    "Outcome",
    "run_episode",
    "RolloutConfig",
    "TrainingExample",
    "EpisodeResult",
    "Observation",
    "Policy",
    "Driver",
    "JsonlWriter",
]
