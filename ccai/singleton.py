"""
A singleton implementation
"""

from typing import Any, Callable, Optional


# pylint: disable=R0903
class Singleton:
    """A simple singleton base class"""

    _instance: Optional[Callable] = None

    def __init__(self) -> None:
        pass

    def __new__(cls, *args: Any, **kwargs: Any) -> Callable:
        """Override __new__ to implement custom instantiation"""
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(  # type: ignore
                cls, *args, **kwargs
            )
        return cls._instance
