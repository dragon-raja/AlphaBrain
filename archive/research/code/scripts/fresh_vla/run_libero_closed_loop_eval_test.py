from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("run_libero_closed_loop_eval.sh")


class ClosedLoopEvalRunnerTest(unittest.TestCase):
    def test_deterministic_reach_receives_requested_split(self) -> None:
        source = SCRIPT.read_text()
        reach_block = source.split(
            'if [ "$EVAL_ONLY" = all ] || [ "$EVAL_ONLY" = reach ]; then', 1
        )[1]

        self.assertIn('--split "$EVAL_SPLIT"', reach_block)
        self.assertIn('payload.get("split") == sys.argv[2]', reach_block)
        self.assertIn('len(rows) == 3 * group_count', reach_block)


if __name__ == "__main__":
    unittest.main()
