from contextlib import contextmanager
import deepspeed
import wandb

class _DummyRun:
    """No-op stand-in for wandb.Run on non-main ranks."""
    def log(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        # Any other method (e.g., watch, save, etc.) becomes a no-op.
        def _no_op(*args, **kwargs):
            pass
        return _no_op


@contextmanager
def ds_wandb(*, main: bool, config=None, **kwargs):
    """
    Context manager that:
      - On rank 0: calls wandb.init(...) and yields the real run; then run.finish().
      - On other ranks: yields a dummy object with .log() etc. that do nothing.
    """

    if not main:
        # Non-main processes (or no wandb installed) just get a dummy object
        yield _DummyRun()
        return

    run = wandb.init(config=config, **kwargs)
    try:
        yield run
    finally:
        run.finish()