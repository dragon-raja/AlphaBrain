"""Test-only override for named boundaries after replacing all real IO with fakes.

Never used by admission tests or runtime code. Unlisted boundaries still reject.
"""
from AlphaBrain.research.dsol.execution import dispatch_policy


def allow_synthetic_boundaries(monkeypatch, *entries):
    original = dispatch_policy.reject_production

    def check(entry):
        if entry not in entries:
            original(entry)

    monkeypatch.setattr(dispatch_policy, 'reject_production', check)
