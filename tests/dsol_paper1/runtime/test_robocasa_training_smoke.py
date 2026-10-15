"""Runtime resource smoke must not launch GPU work via help/default invocation."""

from pathlib import Path
import subprocess
import sys
import unittest


class ResourceProbeBoundaryTests(unittest.TestCase):
    def test_help_without_runtime_imports(self):
        script = Path(__file__).resolve().parents[3] / 'scripts/dsol_paper1/diagnostics/robocasa_training_smoke.py'
        result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--execute', result.stdout)

    def test_execution_requires_explicit_flag(self):
        script = Path(__file__).resolve().parents[3] / 'scripts/dsol_paper1/diagnostics/robocasa_training_smoke.py'
        result = subprocess.run([sys.executable, str(script), '--openpi-root', '/not-accessed',
                                 '--checkpoint', '/not-accessed', '--output', '/not-accessed/result.json'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Explicit --execute', result.stderr)
