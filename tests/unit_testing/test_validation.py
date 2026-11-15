import pytest

from dcm_classifier._validation import ArgumentTypeError, require_type


def test_accepts_the_expected_type():
    require_type("t1w", str, "set_modality")


@pytest.mark.parametrize("caught", [TypeError, ValueError, ArgumentTypeError])
def test_wrong_type_is_caught_as_type_and_value_error(caught):
    with pytest.raises(caught, match="Can only set_modality with a str"):
        require_type(4, str, "set_modality")
