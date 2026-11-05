"""Idempotent process thread configuration for sequential resident runtimes."""
from threading import Lock

_LOCK = Lock()


def configure_threads(torch, *, intra=2, interop=1):
    """Keep exact requested counts without repeating PyTorch's one-shot setter.

    A conflicting late change still raises; errors are never swallowed. Call
    before the first model load, and safely repeat for subsequent leases.
    This function performs no import, model construction or device operation.
    """
    if any(type(n) is not int or n < 1 for n in (intra, interop)):
        raise ValueError('Explicit positive integer thread counts required')
    with _LOCK:
        if torch.get_num_interop_threads() != interop:
            torch.set_num_interop_threads(interop)
        if torch.get_num_threads() != intra:
            torch.set_num_threads(intra)
        if (torch.get_num_interop_threads() != interop
                or torch.get_num_threads() != intra):
            raise RuntimeError('Requested PyTorch thread configuration not applied')
