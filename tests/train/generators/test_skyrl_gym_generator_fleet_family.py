"""Tests for SkyRL generator's per-Fleet-env model_family inject.

Production gap this closes: before this PR, SkyRL's Qwen GRPO runs landed
in the fleet_task env without `model_family` in `env_extras` because the
dataset rows don't carry it. The new per-turn observation scaffold + the
existing canonical-format reject message both depend on model_family;
without it production Qwen runs would silently lose the turn indicator
and the canonical-format reject. The Tinker entrypoint (main_fleet_tinker)
sets model_family explicitly, so the gap only affects SkyRL.

The fix derives family from the generator's model_name, only for Fleet
env classes, and only when extras don't already carry an explicit value.
"""

from __future__ import annotations

from skyrl.train.generators.skyrl_gym_generator import _inject_fleet_model_family


class TestInjectFleetModelFamily:
    def test_qwen_model_into_fleet_env_sets_family(self):
        extras = {}
        _inject_fleet_model_family(extras, "fleet_task", "Qwen/Qwen3.5-9B")
        assert extras["model_family"] == "qwen"

    def test_kimi_model_into_fleet_env_sets_family(self):
        extras = {}
        _inject_fleet_model_family(extras, "fleet_task", "moonshotai/Kimi-K2.6")
        assert extras["model_family"] == "kimi"

    def test_explicit_extras_value_wins(self):
        """Datasets that pre-label model_family must not be overridden."""
        extras = {"model_family": "custom"}
        _inject_fleet_model_family(extras, "fleet_task", "Qwen/Qwen3.5-9B")
        assert extras["model_family"] == "custom"

    def test_unknown_model_name_no_inject(self):
        """Unrecognized model → no inject → env falls through to today's
        no-scaffold behavior. Better silent passthrough than wrong family."""
        extras = {}
        _inject_fleet_model_family(extras, "fleet_task", "some-unknown/model")
        assert "model_family" not in extras

    def test_empty_model_name_no_inject(self):
        extras = {}
        _inject_fleet_model_family(extras, "fleet_task", "")
        assert "model_family" not in extras

    def test_non_fleet_env_class_noop(self):
        """Other env classes (gsm8k, math, etc.) must not get model_family
        appended — they don't read it and adding it is dead weight."""
        extras = {}
        _inject_fleet_model_family(extras, "gsm8k", "Qwen/Qwen3.5-9B")
        assert "model_family" not in extras

    def test_fleet_env_prefix_match(self):
        """Variant Fleet env class names (fleet_env, fleet_task) all match."""
        for env_class in ("fleet_task", "fleet_env", "fleet_browser"):
            extras = {}
            _inject_fleet_model_family(extras, env_class, "Qwen/Qwen3.5-9B")
            assert extras.get("model_family") == "qwen", (
                f"prefix match failed for env_class={env_class}"
            )
