"""Entrypoint wiring for ATOF: default -> emitter installed, opt-out -> untouched.

Uses the real install path end to end (real init_atof, real SkyRLGymGenerator,
the real exp get_generator methods). The only stand-in is the FakeNemo module
from test_atof_events, because the real wheel is linux-only and installed on
training boxes at launch, not a repo dependency.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from integrations.fleet.atof_events import AtofEmitter
from integrations.fleet.tests.test_atof_events import FakeNemo
from integrations.fleet.trace_jobs import FleetTraceWrappedGenerator
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator


@pytest.fixture
def enabled_env(monkeypatch):
    monkeypatch.delenv("SKYRL_ATOF_ENABLED", raising=False)
    monkeypatch.setenv("THESEUS_ATOF_MSK_BROKERS", "b-1:9198")
    monkeypatch.setenv("THESEUS_ATOF_TENANT_ID", "skyrl")
    monkeypatch.setitem(sys.modules, "nemo_relay", FakeNemo())


@pytest.fixture
def disabled_env(monkeypatch):
    monkeypatch.setenv("SKYRL_ATOF_ENABLED", "0")


@pytest.fixture
def cfg(mock_tokenizer_free_generator_cfg, monkeypatch):
    """Config mock with the real fields the exp get_generator methods read."""
    # try_load_processor probes HF for the model path; keep the test offline.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = MagicMock()
    cfg.generator = mock_tokenizer_free_generator_cfg
    cfg.environment.skyrl_gym.max_env_workers = 0
    cfg.trainer.run_name = "run-1"
    cfg.trainer.policy.model.path = "test-org/test-model"
    return cfg


@pytest.fixture
def mock_tokenizer_free_generator_cfg():
    """Minimal GeneratorConfig so a real SkyRLGymGenerator constructs."""
    from skyrl.train.config import GeneratorConfig

    generator_cfg = GeneratorConfig()
    generator_cfg.use_conversation_multi_turn = False
    generator_cfg.chat_template_kwargs = {}
    return generator_cfg


def make_tokenizer():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    tokenizer.eos_token_id = 4
    return tokenizer


def build_exp(exp_cls):
    """Instantiate without __init__: get_generator only uses its arguments."""
    return exp_cls.__new__(exp_cls)


def exp_cases():
    from integrations.fleet.entrypoints.main_eval import FleetEvalExp
    from integrations.fleet.entrypoints.main_fleet import FleetPPOExp
    from integrations.fleet.entrypoints.main_negotiation import NegotiationPPOExp
    from integrations.fleet.entrypoints.main_task_gen import FleetPPOExp as TaskGenExp

    # (exp class, expected entrypoint tag, wrapped in FleetTraceWrappedGenerator)
    return [
        (FleetPPOExp, "main_fleet", True),
        (NegotiationPPOExp, "main_negotiation", True),
        (FleetEvalExp, "main_eval", True),
        (TaskGenExp, "main_task_gen", False),
    ]


@pytest.mark.parametrize("exp_cls,entrypoint,wrapped", exp_cases())
def test_enabled_installs_emitter_on_inner_generator(enabled_env, cfg, exp_cls, entrypoint, wrapped):
    exp = build_exp(exp_cls)
    result = exp.get_generator(cfg, make_tokenizer(), MagicMock())

    inner = result.generator if wrapped else result
    assert isinstance(inner, SkyRLGymGenerator)
    assert isinstance(inner.atof_emitter, AtofEmitter)
    assert inner.atof_emitter._entrypoint == entrypoint
    assert inner.atof_emitter._run_name == "run-1"
    if wrapped:
        assert isinstance(result, FleetTraceWrappedGenerator)
        # The wrapper must not shadow the inner attribute.
        assert "atof_emitter" not in vars(result)


@pytest.mark.parametrize("exp_cls,entrypoint,wrapped", exp_cases())
def test_disabled_leaves_generator_untouched(disabled_env, cfg, exp_cls, entrypoint, wrapped):
    exp = build_exp(exp_cls)
    result = exp.get_generator(cfg, make_tokenizer(), MagicMock())

    inner = result.generator if wrapped else result
    assert inner.atof_emitter is None
