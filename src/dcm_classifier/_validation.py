"""Run-time argument checks for callers that are not type checked."""


class ArgumentTypeError(TypeError, ValueError):
    """An argument has the wrong type.

    It is a ValueError as well, because the setters raised ValueError before.
    """


def require_type(value: object, expected: type, setter: str) -> None:
    """
    Raise ArgumentTypeError unless value is an instance of expected.

    :param value: The argument passed to the setter.
    :param expected: The type the setter accepts.
    :param setter: The setter's name, used in the message.
    :raises ArgumentTypeError: If value is not an instance of expected; it is
        both a TypeError and a ValueError.
    """
    if not isinstance(value, expected):
        msg = f"ERROR: Can only {setter} with a {expected.__name__}. Got {type(value)}."
        raise ArgumentTypeError(msg)
