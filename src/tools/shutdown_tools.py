from tools.code_runner.sandbox import shutdown_sandbox


def shutdown_tools() -> None:
    """Shut down all persistent tool resources."""
    shutdown_sandbox()
