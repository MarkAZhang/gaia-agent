from e2b_code_interpreter import Sandbox

_sandbox: Sandbox | None = None


def get_sandbox() -> Sandbox:
    """Return the persistent sandbox, creating it on first call."""
    global _sandbox
    if _sandbox is None:
        _sandbox = Sandbox.create(timeout=3600)
    return _sandbox


def shutdown_sandbox() -> None:
    """Shut down the persistent sandbox if one exists."""
    global _sandbox
    if _sandbox is not None:
        _sandbox.kill()
        _sandbox = None
