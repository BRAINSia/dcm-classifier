"""Typed front for the deprecation package, which ships without type information."""

from collections.abc import Callable
from typing import Any, cast

from deprecation import deprecated as _untyped_deprecated


def deprecated[F: Callable[..., Any]](**kwargs: Any) -> Callable[[F], F]:
    """
    Mark a function as deprecated; see deprecation.deprecated for the arguments.

    :return: A decorator that returns a function of the same type.
    """
    return cast("Callable[[F], F]", _untyped_deprecated(**kwargs))
