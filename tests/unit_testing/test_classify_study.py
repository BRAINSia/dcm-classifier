import logging

import pytest

from dcm_classifier.classify_study import _LevelForProblemsFormatter


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (logging.INFO, "slice count 12"),
        (logging.WARNING, "WARNING: slice count 12"),
        (logging.ERROR, "ERROR: slice count 12"),
    ],
)
def test_formatter_prefixes_only_problems(level, expected):
    record = logging.LogRecord(
        "dcm_classifier", level, __file__, 1, "slice count %d", (12,), None
    )
    assert _LevelForProblemsFormatter().format(record) == expected
