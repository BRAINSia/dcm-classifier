"""Series whose acquisitions repeat slice positions.

Releases up to 0.9.5rc2 packed such acquisitions into one volume when they
differed only in tags outside GDCM's series grouping (AcquisitionNumber, time).
"""

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from dcm_classifier.study_processing import (
    ProcessOneDicomStudyToVolumesMappingBase,
    _acquisition_time,
)
from dcm_classifier.utility_functions import SliceOrdering

SLICE_SPACING = 4.0
GEOMETRIC = {SliceOrdering.GEOMETRIC, SliceOrdering.GEOMETRIC_SUBVOLUME}


def _write(
    directory: Path,
    acquisitions: list[range],
    acquisition_numbers: list[int] | None = None,
    siemens_bvalues: list[int] | None = None,
    number_each_slice: bool = False,
    sagittal_rank: int | None = None,
    descending_acquisitions: frozenset[int] = frozenset(),
    temporal_positions: list[int] | None = None,
    instance_number=None,
) -> int:
    """Write one series of acquisitions[a] slice ranks; each pixel holds 100 * a + rank."""
    series_uid, study_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    written = 0
    for a, ranks in enumerate(acquisitions):
        for k, rank in enumerate(ranks):
            in_acquisition = len(ranks) - k if a in descending_acquisitions else k + 1
            instance = written + in_acquisition
            if instance_number is not None:
                instance = instance_number(a, rank)
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = pydicom.Dataset()
            ds.file_meta = meta
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
            ds.FrameOfReferenceUID = frame_uid
            ds.Modality = "MR"
            ds.SeriesNumber = 1
            if number_each_slice:
                ds.AcquisitionNumber = written + k + 1
            elif acquisition_numbers is not None:
                ds.AcquisitionNumber = acquisition_numbers[a]
            ds.InstanceNumber = instance
            if temporal_positions is not None:
                ds.TemporalPositionIdentifier = temporal_positions[a]
            ds.ImageType = ["ORIGINAL", "PRIMARY"]
            ds.ImageOrientationPatient = (
                [0, 1, 0, 0, 0, -1] if rank == sagittal_rank else [1, 0, 0, 0, 1, 0]
            )
            ds.ImagePositionPatient = [0.0, 0.0, SLICE_SPACING * rank]
            ds.PixelSpacing = [1.0, 1.0]
            ds.SliceThickness = SLICE_SPACING
            ds.Rows = ds.Columns = 4
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            if siemens_bvalues is not None:
                ds.add_new((0x0019, 0x0010), "LO", "SIEMENS MR HEADER")
                ds.add_new((0x0019, 0x100C), "IS", str(siemens_bvalues[a]))
            ds.PixelData = np.full((4, 4), 100 * a + rank, np.uint16).tobytes()
            ds.save_as(directory / f"IM{instance:04d}.dcm", enforce_file_format=True)
        written += len(ranks)
    return written


def _volumes(directory: Path, **kwargs):
    study = ProcessOneDicomStudyToVolumesMappingBase(directory, **kwargs)
    (series,) = study.series_dictionary.values()
    return series.get_volume_list()


def _content(volume) -> list[tuple[int, int]]:
    """(acquisition, rank) of each slice, in the volume's file order."""
    values = [
        int(pydicom.dcmread(f).pixel_array[0, 0])
        for f in volume.get_one_volume_dcm_filenames()
    ]
    return [divmod(v, 100) for v in values]


def _assert_every_file_once(volumes, n_files: int) -> None:
    names = Counter(str(f) for v in volumes for f in v.get_one_volume_dcm_filenames())
    assert len(names) == n_files
    assert set(names.values()) == {1}


def _assert_one_acquisition_per_volume(volumes, sizes: list[int]) -> None:
    assert [len(v.get_one_volume_dcm_filenames()) for v in volumes] == sizes
    for index, volume in enumerate(volumes):
        content = _content(volume)
        assert {a for a, _ in content} == {index}
        assert [r for _, r in content] == sorted(r for _, r in content)
        assert volume.get_slice_ordering() in GEOMETRIC


def test_time_series_repeats_are_separate_volumes(tmp_path):
    n = _write(tmp_path, [range(6)] * 4, acquisition_numbers=[1, 2, 3, 4])
    volumes = _volumes(tmp_path)
    _assert_every_file_once(volumes, n)
    _assert_one_acquisition_per_volume(volumes, [6, 6, 6, 6])


def test_repeated_diffusion_encodings_are_separate_volumes(tmp_path):
    n = _write(
        tmp_path,
        [range(6)] * 4,
        acquisition_numbers=[1, 2, 3, 4],
        siemens_bvalues=[0, 1000, 0, 1000],
    )
    volumes = _volumes(tmp_path)
    _assert_every_file_once(volumes, n)
    assert sorted(len(v.get_one_volume_dcm_filenames()) for v in volumes) == [6] * 4
    for volume in volumes:
        assert len({a for a, _ in _content(volume)}) == 1


def test_incomplete_last_acquisition_keeps_every_file(tmp_path):
    n = _write(
        tmp_path,
        [range(6), range(6), range(6), range(3)],
        acquisition_numbers=[1, 2, 3, 4],
    )
    volumes = _volumes(tmp_path)
    _assert_every_file_once(volumes, n)
    _assert_one_acquisition_per_volume(volumes, [6, 6, 6, 3])


def test_shifted_acquisitions_are_split_by_acquisition_number(tmp_path):
    n = _write(tmp_path, [range(6), range(1, 7)], acquisition_numbers=[1, 2])
    volumes = _volumes(tmp_path)
    _assert_every_file_once(volumes, n)
    _assert_one_acquisition_per_volume(volumes, [6, 6])


def test_partly_overlapping_acquisitions_are_not_packed(tmp_path):
    n = _write(
        tmp_path, [range(6), range(6), range(3, 9)], acquisition_numbers=[1, 2, 3]
    )
    volumes = _volumes(tmp_path)
    _assert_every_file_once(volumes, n)
    _assert_one_acquisition_per_volume(volumes, [6, 6, 6])


def test_mosaic_like_single_position_repeats(tmp_path):
    n = _write(tmp_path, [range(1)] * 4, acquisition_numbers=[1, 2, 3, 4])
    volumes = _volumes(tmp_path)
    _assert_every_file_once(volumes, n)
    _assert_one_acquisition_per_volume(volumes, [1, 1, 1, 1])


def test_acquisition_number_per_slice_does_not_split_slices(tmp_path):
    n = _write(tmp_path, [range(6)] * 2, number_each_slice=True)
    volumes = _volumes(tmp_path)
    _assert_every_file_once(volumes, n)
    _assert_one_acquisition_per_volume(volumes, [6, 6])


@pytest.mark.parametrize("acquisition_numbers", [None, [1, 1, 1, 1]])
def test_uneven_repeats_without_acquisition_identity_are_flagged(
    tmp_path, acquisition_numbers
):
    n = _write(
        tmp_path,
        [range(6), range(6), range(6), range(3)],
        acquisition_numbers=acquisition_numbers,
    )
    volumes = _volumes(tmp_path, allow_fallback_ordering=True)
    _assert_every_file_once(volumes, n)
    assert {v.get_slice_ordering() for v in volumes} == {
        SliceOrdering.FALLBACK_REPEATED_POSITIONS
    }
    with pytest.raises(ValueError, match="fallback_repeated_positions"):
        _volumes(tmp_path)


def test_chained_overlapping_acquisitions_are_split(tmp_path):
    n = _write(
        tmp_path, [range(6), range(2, 8), range(6, 12)], acquisition_numbers=[1, 2, 3]
    )
    volumes = _volumes(tmp_path)
    _assert_every_file_once(volumes, n)
    _assert_one_acquisition_per_volume(volumes, [6, 6, 6])


@pytest.mark.parametrize("corrupt", ["IM0002.dcm", "IM0005.dcm"])
def test_unparsable_acquisition_number_is_refused(tmp_path, corrupt):
    n = _write(tmp_path, [range(3)] * 2, acquisition_numbers=[12345, 23456])
    bad = tmp_path / corrupt
    bad.write_bytes(
        bad.read_bytes().replace(b"12345 ", b"xx345 ").replace(b"23456 ", b"xx456 ")
    )
    with pytest.raises(ValueError, match="fallback_repeated_positions"):
        _volumes(tmp_path)
    volumes = _volumes(tmp_path, allow_fallback_ordering=True)
    _assert_every_file_once(volumes, n)
    assert {v.get_slice_ordering() for v in volumes} == {
        SliceOrdering.FALLBACK_REPEATED_POSITIONS
    }


def test_interleaved_halves_of_each_timepoint_stay_together(tmp_path):
    halves = [range(0, 6, 2), range(1, 6, 2)]
    n = _write(tmp_path, halves * 2, acquisition_numbers=[1, 2, 3, 4])
    volumes = _volumes(tmp_path)
    _assert_every_file_once(volumes, n)
    assert [len(v.get_one_volume_dcm_filenames()) for v in volumes] == [6, 6]
    for timepoint, volume in enumerate(volumes):
        content = _content(volume)
        assert {a for a, _ in content} == {2 * timepoint, 2 * timepoint + 1}
        assert [r for _, r in content] == list(range(6))
        assert volume.get_slice_ordering() in GEOMETRIC


def test_fallback_sub_volume_is_read_in_instance_number_order(tmp_path):
    n = _write(
        tmp_path,
        [range(6)] * 2,
        acquisition_numbers=[1, 2],
        sagittal_rank=2,
        descending_acquisitions=frozenset({1}),
    )
    volumes = _volumes(tmp_path, allow_fallback_ordering=True)
    _assert_every_file_once(volumes, n)
    assert len(volumes) == 2
    for volume in volumes:
        assert volume.get_slice_ordering() == SliceOrdering.FALLBACK_INSTANCE_NUMBER
        numbers = [
            int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber)
            for f in volume.get_one_volume_dcm_filenames()
        ]
        assert numbers == sorted(numbers)


def test_temporal_position_separates_repeats_numbered_out_of_order(tmp_path):
    n = _write(
        tmp_path,
        [range(3)] * 2,
        temporal_positions=[1, 2],
        instance_number=lambda a, rank: 2 * rank + (a if rank % 2 == 0 else 1 - a) + 1,
    )
    volumes = _volumes(tmp_path)
    _assert_every_file_once(volumes, n)
    _assert_one_acquisition_per_volume(volumes, [3, 3])


def test_acquisitions_with_mixed_orientations_are_refused(tmp_path):
    n = _write(
        tmp_path,
        [range(6), range(2, 8), range(6, 12)],
        acquisition_numbers=[1, 2, 3],
        sagittal_rank=0,
    )
    with pytest.raises(ValueError, match="fallback"):
        _volumes(tmp_path)
    volumes = _volumes(tmp_path, allow_fallback_ordering=True)
    _assert_every_file_once(volumes, n)


@pytest.mark.parametrize("temporal_positions", [None, [1, 2]])
def test_repeats_numbered_out_of_order_without_a_full_tag_are_refused(
    tmp_path, temporal_positions
):
    n = _write(
        tmp_path,
        [range(4)] * 2,
        acquisition_numbers=[7, 7],
        temporal_positions=temporal_positions,
        instance_number=lambda a, rank: 2 * rank + (a if rank % 2 == 0 else 1 - a) + 1,
    )
    if temporal_positions is not None:
        partial = tmp_path / "IM0001.dcm"
        ds = pydicom.dcmread(partial)
        del ds.TemporalPositionIdentifier
        ds.save_as(partial)
    with pytest.raises(ValueError, match="fallback_repeated_positions"):
        _volumes(tmp_path)
    volumes = _volumes(tmp_path, allow_fallback_ordering=True)
    _assert_every_file_once(volumes, n)


def _time_two_acquisitions(directory: Path, odd_file: str, odd_tags: dict) -> None:
    """Acquisition a at 1a:00:00; odd_file gets odd_tags (None deletes a tag)."""
    for path in sorted(directory.glob("*.dcm")):
        ds = pydicom.dcmread(path)
        acquisition = int(ds.pixel_array[0, 0]) // 100
        ds.AcquisitionDate = "20260101"
        ds.AcquisitionTime = f"{10 + acquisition:02d}0000"
        if path.name == odd_file:
            for keyword, value in odd_tags.items():
                if value is None:
                    delattr(ds, keyword)
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    setattr(ds, keyword, value)
        ds.save_as(path)


@pytest.mark.parametrize(
    "odd_tags",
    [
        {"AcquisitionDate": None},
        {"AcquisitionDate": "2026.01.01"},
        {"AcquisitionTime": "11:00:00"},
        {"AcquisitionDateTime": "20260101110000.000000"},
        {"AcquisitionTime": None},
        {"AcquisitionTime": ""},
    ],
)
def test_repeats_with_an_unusable_acquisition_time_are_ordered_by_instance(
    tmp_path, odd_tags
):
    n = _write(
        tmp_path,
        [range(4)] * 2,
        acquisition_numbers=[7, 7],
        instance_number=lambda a, rank: 4 * a + 4 - rank,
    )
    _time_two_acquisitions(tmp_path, "IM0008.dcm", odd_tags)
    volumes = _volumes(tmp_path)
    _assert_every_file_once(volumes, n)
    _assert_one_acquisition_per_volume(volumes, [4, 4])


def test_repeats_with_a_missing_instance_number_are_labeled_fallback(tmp_path):
    n = _write(tmp_path, [range(4)] * 2, acquisition_numbers=[7, 7])
    odd = tmp_path / "IM0008.dcm"
    ds = pydicom.dcmread(odd)
    del ds.InstanceNumber
    ds.save_as(odd)
    # ITK's GDCM series sorter can crash on this series, so call the splitter directly.
    study = ProcessOneDicomStudyToVolumesMappingBase.__new__(
        ProcessOneDicomStudyToVolumesMappingBase
    )
    study.raise_error_on_failure = False
    volumes = study._identify_and_split_sub_volumes_pydicom(
        [str(f) for f in sorted(tmp_path.glob("*.dcm"))]
    )
    _assert_every_file_once(volumes, n)
    assert {v.get_slice_ordering() for v in volumes} == {
        SliceOrdering.FALLBACK_REPEATED_POSITIONS
    }


@pytest.mark.parametrize(
    "clock",
    [
        [
            "235950",
            "235955",
            "000000",
            "000005",
            "000500",
            "000500",
            "000500",
            "000500",
        ],
        [
            "100000",
            "100000",
            "100000",
            "100000",
            "103000",
            "103000",
            "103000",
            "103000",
        ],
    ],
    ids=["midnight-without-date-change", "offset-on-some-files"],
)
def test_times_that_contradict_instance_order_never_mix(tmp_path, clock):
    n = _write(tmp_path, [range(4)] * 2, acquisition_numbers=[7, 7])
    for index, path in enumerate(sorted(tmp_path.glob("*.dcm"))):
        ds = pydicom.dcmread(path)
        ds.AcquisitionDate = "20260101"
        ds.AcquisitionTime = clock[index]
        if clock[0] == "100000" and index in (2, 3):
            del ds.AcquisitionDate
            del ds.AcquisitionTime
            ds.AcquisitionDateTime = "20260101150000+0000"
        ds.save_as(path)
    volumes = _volumes(tmp_path, allow_fallback_ordering=True)
    _assert_every_file_once(volumes, n)
    for volume in volumes:
        if volume.get_slice_ordering() != SliceOrdering.FALLBACK_REPEATED_POSITIONS:
            assert len({a for a, _ in _content(volume)}) == 1


def test_wrong_times_with_slice_by_slice_numbering_never_mix(tmp_path):
    n = _write(
        tmp_path,
        [range(4)] * 2,
        acquisition_numbers=[7, 7],
        instance_number=lambda a, rank: 2 * rank + a + 1,
    )
    clock = {
        0: ["235950", "235955", "000005", "000006"],
        1: ["000500", "000510", "000520", "000530"],
    }
    for path in sorted(tmp_path.glob("*.dcm")):
        ds = pydicom.dcmread(path)
        acquisition, rank = divmod(int(ds.pixel_array[0, 0]), 100)
        ds.AcquisitionDate = "20260101"
        ds.AcquisitionTime = clock[acquisition][rank]
        ds.save_as(path)
    volumes = _volumes(tmp_path, allow_fallback_ordering=True)
    _assert_every_file_once(volumes, n)
    for volume in volumes:
        if volume.get_slice_ordering() != SliceOrdering.FALLBACK_REPEATED_POSITIONS:
            assert len({a for a, _ in _content(volume)}) == 1


def _clock_repeats(directory: Path, clock) -> None:
    for path in sorted(directory.glob("*.dcm")):
        ds = pydicom.dcmread(path)
        acquisition, rank = divmod(int(ds.pixel_array[0, 0]), 100)
        ds.AcquisitionDate = "20260101"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds.AcquisitionTime = clock(acquisition, rank)
        ds.save_as(path)


def test_one_wrong_time_with_tied_instance_numbers_is_refused(tmp_path):
    n = _write(
        tmp_path,
        [range(4)] * 2,
        acquisition_numbers=[7, 7],
        instance_number=lambda a, rank: rank + 1 + 100 * a,
    )
    for path in sorted(tmp_path.glob("*.dcm")):
        ds = pydicom.dcmread(path)
        ds.InstanceNumber = int(ds.InstanceNumber) % 100
        ds.save_as(path)
    _clock_repeats(
        tmp_path,
        lambda a, rank: "095900" if (a, rank) == (1, 3) else f"10{a:02d}0{rank}",
    )
    with pytest.raises(ValueError, match="fallback_repeated_positions"):
        _volumes(tmp_path)
    volumes = _volumes(tmp_path, allow_fallback_ordering=True)
    _assert_every_file_once(volumes, n)


def test_one_wrong_instance_number_contradicted_by_times_is_refused(tmp_path):
    n = _write(
        tmp_path,
        [range(4)] * 2,
        acquisition_numbers=[7, 7],
        instance_number=lambda a, rank: 0 if (a, rank) == (1, 3) else 4 * a + rank + 1,
    )
    _clock_repeats(tmp_path, lambda a, rank: f"10{a:02d}0{rank}")
    with pytest.raises(ValueError, match="fallback_repeated_positions"):
        _volumes(tmp_path)
    volumes = _volumes(tmp_path, allow_fallback_ordering=True)
    _assert_every_file_once(volumes, n)


def test_wrong_instance_number_is_caught_by_the_times_that_parse(tmp_path):
    n = _write(
        tmp_path,
        [range(4)] * 2,
        acquisition_numbers=[7, 7],
        instance_number=lambda a, rank: 9 if (a, rank) == (0, 3) else 4 * a + 4 - rank,
    )
    _clock_repeats(
        tmp_path, lambda a, rank: "xx" if (a, rank) == (1, 0) else f"10{a:02d}0{rank}"
    )
    with pytest.raises(ValueError, match="fallback_repeated_positions"):
        _volumes(tmp_path)
    volumes = _volumes(tmp_path, allow_fallback_ordering=True)
    _assert_every_file_once(volumes, n)


def test_temporal_position_contradicted_by_acquisition_number_is_refused(tmp_path):
    n = _write(
        tmp_path,
        [range(4), range(4, 8), range(8)],
        acquisition_numbers=[1, 2, 3],
        temporal_positions=[1, 2, 3],
    )
    odd = tmp_path / "IM0004.dcm"
    ds = pydicom.dcmread(odd)
    ds.TemporalPositionIdentifier = 2
    ds.save_as(odd)
    with pytest.raises(ValueError, match="fallback_repeated_positions"):
        _volumes(tmp_path)
    volumes = _volumes(tmp_path, allow_fallback_ordering=True)
    _assert_every_file_once(volumes, n)


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (
            {"AcquisitionDateTime": "20260101103045.25+0100"},
            "2026-01-01 10:30:45.250000",
        ),
        ({"AcquisitionDateTime": "20260101103045+1400"}, "2026-01-01 10:30:45"),
        ({"AcquisitionDateTime": "20260101103045+1430"}, None),
        ({"AcquisitionDateTime": "20260101103045-2400"}, None),
        ({"AcquisitionDateTime": "20260101"}, None),
        ({"AcquisitionDateTime": "2026010110.5"}, None),
        (
            {"AcquisitionDate": "20260101", "AcquisitionTime": "1030"},
            "2026-01-01 10:30:00",
        ),
        ({"AcquisitionDate": "2026010111", "AcquisitionTime": "30"}, None),
        ({"AcquisitionDate": "20260101"}, None),
    ],
)
def test_acquisition_time_accepts_only_valid_dicom_values(tags, expected):
    ds = pydicom.Dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for keyword, value in tags.items():
            setattr(ds, keyword, value)
    parsed = _acquisition_time(ds)
    assert (None if parsed is None else str(parsed)) == expected

