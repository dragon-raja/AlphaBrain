"""Closed historical CLI. Implementation is retained in the maintenance archive."""
from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production


def run(root):
    reject_production('benchmarks/LIBERO/baseline_dispatch.py:run')


if __name__ == '__main__':
    run(None)
