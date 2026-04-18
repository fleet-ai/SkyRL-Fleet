#!/usr/bin/env python3
"""Gym-Anything server smoke test.

Verifies that environments boot correctly and produce non-blank, responsive
screenshots before committing GPU-hours to training/eval.

Usage:
    python scripts/gym-anything-smoke-test.py http://<server-ip>:5000
    python scripts/gym-anything-smoke-test.py http://<server-ip>:5000 --env-dir-prefix /home/gcpuser/gym-anything/benchmarks/cua_world/environments
"""

import argparse
import base64
import hashlib
import json
import sys
import time

import requests

TEST_ENVS = [
    {"env_name": "stellarium_env", "task_id": "observe_solar_eclipse"},
    {"env_name": "libreoffice_calc_env", "task_id": "accessible_venue_eval"},
    {"env_name": "firefox_env", "task_id": "a11y_compliance_audit"},
]

MIN_SCREENSHOT_CHARS = 100_000  # ~75KB decoded. Blank desktops are <10KB.


def check_health(server_url: str) -> bool:
    try:
        r = requests.get(f"{server_url}/health", timeout=10)
        data = r.json()
        if data.get("healthy_workers", 0) == 0:
            print(f"  FAIL: no healthy workers ({data})")
            return False
        print(f"  OK: {data['healthy_workers']} workers, {data['total_capacity']} capacity")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_env(server_url: str, env_dir: str, task_id: str, env_name: str, timeout: int) -> bool:
    env_id = None
    try:
        # 1. Create
        r = requests.post(
            f"{server_url}/envs/create",
            json={"env_dir": env_dir, "task_id": task_id},
            timeout=60,
        )
        r.raise_for_status()
        env_id = r.json().get("env_id")
        if not env_id:
            print(f"  FAIL: no env_id ({r.json()})")
            return False

        # 2. Reset
        r = requests.post(
            f"{server_url}/envs/{env_id}/reset",
            json={"use_cache": True, "cache_level": "post_start"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            print(f"  FAIL: reset error: {data['error'][:100]}")
            return False

        # 3. Check screenshot is present and non-blank
        obs = data.get("observation") or {}
        screen = obs.get("screen") or {}
        png_b64 = screen.get("png_b64", "")
        if len(png_b64) < MIN_SCREENSHOT_CHARS:
            print(f"  FAIL: screenshot too small ({len(png_b64)} chars, need >{MIN_SCREENSHOT_CHARS})")
            return False
        hash1 = hashlib.sha256(base64.b64decode(png_b64)).hexdigest()[:16]
        print(f"  Screenshot OK ({len(png_b64)} chars, hash={hash1})")

        # 4. Click center of screen
        r = requests.post(
            f"{server_url}/envs/{env_id}/step",
            json={"actions": [{"mouse": {"left_click": [960, 540]}}]},
            timeout=120,
        )
        r.raise_for_status()
        step_data = r.json()
        step_obs = (step_data.get("observation") or {}).get("screen") or {}
        png_b64_2 = step_obs.get("png_b64", "")
        if len(png_b64_2) < MIN_SCREENSHOT_CHARS:
            print(f"  FAIL: post-action screenshot too small ({len(png_b64_2)} chars)")
            return False
        hash2 = hashlib.sha256(base64.b64decode(png_b64_2)).hexdigest()[:16]

        # 5. Check desktop responded (screenshots differ)
        if hash1 == hash2:
            print(f"  WARN: screenshots identical after click (hash={hash1}). Desktop may not be responsive.")
            # Not a hard failure — some apps don't change on a center click
        else:
            print(f"  Desktop responsive (hash {hash1} -> {hash2})")

        print(f"  PASS")
        return True

    except requests.exceptions.Timeout:
        print(f"  FAIL: timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False
    finally:
        if env_id:
            try:
                requests.post(f"{server_url}/envs/{env_id}/close", timeout=10)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Gym-Anything server smoke test")
    parser.add_argument("server_url", help="e.g. http://10.0.0.1:5000")
    parser.add_argument(
        "--env-dir-prefix",
        default=None,
        help="Absolute prefix for env_dir on remote server. Auto-detected if not set.",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Per-env timeout (seconds)")
    args = parser.parse_args()

    server_url = args.server_url.rstrip("/")

    print(f"=== Gym-Anything Smoke Test ===")
    print(f"Server: {server_url}")
    print()

    # Health check
    print("1. Health check")
    if not check_health(server_url):
        sys.exit(1)
    print()

    # Auto-detect env_dir prefix if not specified
    prefix = args.env_dir_prefix
    if not prefix:
        # Try common paths
        for candidate in [
            "/home/gcpuser/gym-anything/benchmarks/cua_world/environments",
            "/root/gym-anything/benchmarks/cua_world/environments",
        ]:
            try:
                r = requests.post(
                    f"{server_url}/envs/create",
                    json={"env_dir": f"{candidate}/stellarium_env", "task_id": "observe_solar_eclipse"},
                    timeout=30,
                )
                if r.ok and r.json().get("env_id"):
                    env_id = r.json()["env_id"]
                    requests.post(f"{server_url}/envs/{env_id}/close", timeout=10)
                    prefix = candidate
                    break
            except Exception:
                continue
        if not prefix:
            print("FAIL: could not auto-detect env_dir prefix. Use --env-dir-prefix.")
            sys.exit(1)
    print(f"Env dir prefix: {prefix}")
    print()

    # Test environments
    passed = 0
    failed = 0
    for i, env in enumerate(TEST_ENVS, 1):
        env_dir = f"{prefix}/{env['env_name']}"
        print(f"2.{i}. Testing {env['env_name']}/{env['task_id']}")
        if test_env(server_url, env_dir, env["task_id"], env["env_name"], args.timeout):
            passed += 1
        else:
            failed += 1
        print()

    # Summary
    print(f"=== Results: {passed} passed, {failed} failed ===")
    if failed > 0:
        print("SMOKE TEST FAILED — do not proceed with eval/training.")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED — server is ready.")
        sys.exit(0)


if __name__ == "__main__":
    main()
