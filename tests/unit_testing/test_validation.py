import pytest

from dcm_classifier._validation import require_type


def test_accepts_the_expected_type():
    require_type("t1w", str, "set_modality")


def test_wrong_type_raises_type_error():
    with pytest.raises(TypeError, match="Can only set_modality with a str"):
        require_type(4, str, "set_modality")


def test_wrong_type_is_not_a_value_error():
    with pytest.raises(TypeError) as excinfo:
        require_type(4, str, "set_modality")
    assert not isinstance(excinfo.value, ValueError)
