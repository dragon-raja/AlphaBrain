import os
from pathlib import Path
import subprocess
import sys
import unittest


class RoboCasaProbeSafetyTests(unittest.TestCase):
    def test_cli_is_safe_before_execute(self):
        root = Path(__file__).resolve().parents[3]
        script = root / 'scripts/dsol_paper1/runtime/robocasa_policy_probe.py'
        for arguments, expected in [(['--help'], 0), (['--checkpoint', '/nonexistent', '--output', '/nonexistent/result.json'], 2)]:
            result = subprocess.run([sys.executable, str(script), *arguments],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, expected, result.stderr)
            self.assertNotIn('robosuite', result.stderr)

    def test_partial_reference_arguments_fail_before_gpu_import(self):
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run([sys.executable,
            str(root / 'scripts/dsol_paper1/runtime/robocasa_policy_probe.py'),
            '--checkpoint', '/nonexistent', '--output', '/nonexistent/result.json',
            '--reference-plan', '/nonexistent/plan.json', '--execute'],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('required together', result.stderr)
        self.assertNotIn('robosuite', result.stderr)

    def test_experimental_renderer_requires_full_recorded_diagnostic(self):
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run([sys.executable,
            str(root / 'scripts/dsol_paper1/runtime/robocasa_policy_probe.py'),
            '--checkpoint', '/nonexistent', '--output', '/nonexistent/result.json',
            '--diagnostic-msaa-samples', '0', '--execute'],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('full rollout and replay', result.stderr)
        self.assertNotIn('robosuite', result.stderr)
