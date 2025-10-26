"""smith package - tiny demo utilities."""

__version__ = "0.0.1"

def greet(name: str) -> str:
    """Return a simple greeting for `name`.

    This is a tiny example function to demonstrate package behavior.
    """
    return f"Hello, {name}"

__all__ = ["__version__", "greet"]
