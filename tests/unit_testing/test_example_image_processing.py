import importlib.util
from pathlib import Path

script_path: Path = (
    Path(__file__).parents[2] / "scripts" / "example_image_processing.py"
)
spec = importlib.util.spec_from_file_location("example_image_processing", script_path)
example_image_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example_image_processing)
rglob_for_singular_result = example_image_processing.rglob_for_singular_result
slugify = example_image_processing.slugify

relative_testing_data_path: Path = Path(__file__).parent.parent / "testing_data"


def test_rglob_for_singular_result():
    single_file_dir = (
        relative_testing_data_path / "dummy_directory" / "dir_with_one_file"
    )
    single_file = rglob_for_singular_result(single_file_dir, "*.file", "f")
    assert single_file is not None


def test_slugify():
    # TODO: Test should have upper, lower, and unicode characters included in all input test strings
    # TODO: Something like:
    # smiley_unicode = u'\U0001f604'.encode('unicode-escape'
    # f"""start_UPPER_lower_!_@_#_"_'_:_unicode_{smiley_unicode}_end"""
    # NOTE: The same inpute string can be used for all test cases,
    #       there is no need to have multiple input strings
    test_strings = [
        "dwi series number 006 index 01",
        "DWI-series-number-006-index-01",
        "dwi-series-nùmber-006-index-01",
    ]
    results = []

    results.append(slugify(test_strings[0]))
    results.append(slugify(test_strings[1], allow_uppercase=True))
    results.append(slugify(test_strings[2], allow_unicode=True))
    #
    # TODO: results.append(slugify(test_strings[2], allow_uppercase=False))
    # TODO: results.append(slugify(test_strings[4], allow_unicode=False))
    #
    # TODO: results.append(slugify(test_strings[5], allow_uppercase=True, allow_unicode=True))
    # TODO: results.append(slugify(test_strings[6], allow_uppercase=False, allow_unicode=True))
    # TODO: results.append(slugify(test_strings[7], allow_uppercase=True, allow_unicode=False))
    # TODO: results.append(slugify(test_strings[8], allow_uppercase=False, allow_unicode=False))

    assert results == [
        "dwi-series-number-006-index-01",
        "DWI-series-number-006-index-01",
        "dwi-series-nùmber-006-index-01",
    ]
