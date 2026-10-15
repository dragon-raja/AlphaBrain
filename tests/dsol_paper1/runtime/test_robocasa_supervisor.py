import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SupervisorTests(unittest.TestCase):
    def test_records_abnormal_exit(self):
        script = Path(__file__).resolve().parents[3] / 'scripts/dsol_paper1/operations/supervise_robocasa_job.py'
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'run'
            result = subprocess.run([sys.executable, str(script), '--output', str(output),
                '--execute', '--', sys.executable, '-c', 'raise SystemExit(7)'], capture_output=True,
                env=dict(os.environ, PYTHONPATH=str(script.parents[3])))
            self.assertEqual(result.returncode, 7, result.stderr)
            record = json.loads((output / 'status.json').read_text())
            self.assertEqual(record['status'], 'ERROR')
            self.assertEqual(record['returncode'], 7)
