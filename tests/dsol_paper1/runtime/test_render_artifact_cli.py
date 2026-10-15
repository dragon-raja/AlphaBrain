"""Pixel diagnostics never start GPU work through help or implicit execution."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('name,arguments', [
    ('robocasa_render_artifacts.py', ['--snapshot', '/not-accessed']),
    ('robocasa_pixel_action_sensitivity.py', ['--checkpoint', '/not-accessed', '--inputs', '/not-accessed']),
])
def test_explicit_execution(name, arguments):
    script = Path(__file__).resolve().parents[3] / 'scripts/dsol_paper1/diagnostics' / name
    help_result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True, timeout=10)
    assert help_result.returncode == 0
    assert '--execute' in help_result.stdout
    result = subprocess.run([sys.executable, str(script), *arguments, '--output', '/not-accessed'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'Explicit --execute required' in result.stderr


def test_reference_queue_requires_execute():
    script = Path(__file__).resolve().parents[3] / 'scripts/dsol_paper1/operations/refresh_robocasa_references.py'
    result = subprocess.run([sys.executable, str(script), 'run', '--root', '/not-accessed'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'Explicit --execute required' in result.stderr
