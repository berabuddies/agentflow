from __future__ import annotations

import subprocess
from pathlib import Path


def test_verify_async_codex_script_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "verify_async_codex.sh"

    assert script_path.exists()

    completed = subprocess.run(
        ["bash", str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "PR11" in completed.stdout
    assert "PR12" in completed.stdout
    assert "OPENAI_API_KEY" in completed.stdout


def test_verify_async_codex_script_sets_local_git_identity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "verify_async_codex.sh"

    script_text = script_path.read_text(encoding="utf-8")

    assert 'user.name="AgentFlow Verify"' in script_text
    assert 'user.email="verify@example.invalid"' in script_text
