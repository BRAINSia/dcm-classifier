from pathlib import Path

import itk
import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase
from dcm_classifier.utility_functions import SliceOrdering

ROWS, COLUMNS, SLICES = 4, 5, 6
PIXEL_SPACING = (0.9, 0.8)
SLICE_SPACING = 2.5

AXIAL = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
_c, _s = np.cos(np.radians(20)), np.sin(np.radians(20))
OBLIQUE = ((1.0, 0.0, 0.0), (0.0, _c, _s))
SAGITTAL = ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
FALLBACKS = {SliceOrdering.FALLBACK_INSTANCE_NUMBER, SliceOrdering.FALLBACK_FILENAME}


def _write_series(
    directory: Path,
    orientation=OBLIQUE,
    descending_instance_numbers: bool = False,
    orientation_jitter: float = 0.0,
    odd_plane_rank: int | None = None,
    volumes: int = 1,
) -> None:
    """Write a series; each slice's pixels hold 100 * volume + its 1-based position rank."""
    row, column = np.asarray(orientation[0]), np.asarray(orientation[1])
    normal = np.cross(row, column)
    origin = np.array([-10.0, 20.0, 30.0])
    series_uid, study_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    for volume in range(volumes):
        for rank in range(SLICES):
            position = origin + rank * SLICE_SPACING * normal
            in_volume = SLICES - rank if descending_instance_numbers else rank + 1
            instance = volume * SLICES + in_volume
            cosines = [*row, *column]
            if rank == odd_plane_rank:
                cosines = [*SAGITTAL[0], *SAGITTAL[1]]
            # Scanners round the cosines differently from slice to slice.
            jitter = orientation_jitter * (rank % 3)
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = pydicom.Dataset()
            ds.file_meta = meta
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.FrameOfReferenceUID = frame_uid
            ds.Modality = "MR"
            ds.SeriesNumber = 1
            ds.AcquisitionNumber = volume + 1
            ds.InstanceNumber = instance
            ds.ImageOrientationPatient = [
                f"{c + jitter:.10f}" if c else "0" for c in cosines
            ]
            ds.ImagePositionPatient = position.tolist()
            ds.PixelSpacing = list(PIXEL_SPACING)
            ds.SliceThickness = SLICE_SPACING
            ds.Rows, ds.Columns = ROWS, COLUMNS
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            value = 100 * volume + rank + 1
            ds.PixelData = np.full((ROWS, COLUMNS), value, np.uint16).tobytes()
            ds.save_as(directory / f"IM{instance:04d}.dcm", enforce_file_format=True)


def _volumes(directory: Path, **kwargs):
    study = ProcessOneDicomStudyToVolumesMappingBase(directory, **kwargs)
    (series,) = study.series_dictionary.values()
    return series, series.get_volume_list()


def _slice_values(volume) -> list[int]:
    return [int(v) for v in itk.array_view_from_image(volume.get_itk_image())[:, 0, 0]]


def _assert_in_position_order(volume) -> None:
    image = volume.get_itk_image()
    assert np.linalg.det(itk.array_from_matrix(image.GetDirection())) > 0
    values = _slice_values(volume)
    assert [v % 100 for v in values] == list(range(1, SLICES + 1))
    for dcm_file in volume.get_one_volume_dcm_filenames():
        ds = pydicom.dcmread(dcm_file)
        k = int(ds.pixel_array[0, 0]) % 100 - 1
        point = image.TransformIndexToPhysicalPoint([0, 0, k])
        expected = [float(x) for x in ds.ImagePositionPatient]
        np.testing.assert_allclose(point, expected, atol=1e-3)


def _assert_in_instance_number_order(volume) -> None:
    instance_numbers = [
        int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber)
        for f in volume.get_one_volume_dcm_filenames()
    ]
    assert instance_numbers == sorted(instance_numbers)
    by_instance = {
        int(pydicom.dcmread(f).InstanceNumber): int(
            pydicom.dcmread(f).pixel_array[0, 0]
        )
        for f in volume.get_one_volume_dcm_filenames()
    }
    assert _slice_values(volume) == [by_instance[n] for n in sorted(by_instance)]


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("orientation", [AXIAL, OBLIQUE], ids=["axial", "oblique"])
def test_well_formed_series_is_ordered_geometrically(tmp_path, orientation, descending):
    _write_series(tmp_path, orientation, descending_instance_numbers=descending)

    series, (volume,) = _volumes(tmp_path)

    assert volume.get_slice_ordering() == SliceOrdering.GEOMETRIC
    assert series.get_slice_orderings() == {SliceOrdering.GEOMETRIC}
    _assert_in_position_order(volume)


def test_rounded_orientations_report_how_they_were_ordered(tmp_path):
    _write_series(tmp_path, descending_instance_numbers=True, orientation_jitter=1.5e-8)

    _, (volume,) = _volumes(tmp_path, allow_fallback_ordering=True)

    if volume.get_slice_ordering() == SliceOrdering.GEOMETRIC:
        _assert_in_position_order(volume)
    else:
        # ITK 6 before InsightSoftwareConsortium/ITK#7177 cannot sort these.
        assert volume.get_slice_ordering() == SliceOrdering.FALLBACK_INSTANCE_NUMBER
        _assert_in_instance_number_order(volume)


def test_slice_in_another_plane_is_a_fallback(tmp_path):
    _write_series(tmp_path, odd_plane_rank=2)

    _, (volume,) = _volumes(tmp_path, allow_fallback_ordering=True)

    assert volume.get_slice_ordering() == SliceOrdering.FALLBACK_INSTANCE_NUMBER
    _assert_in_instance_number_order(volume)
    assert volume.get_primary_volume_info(0)["SliceOrdering"] == (
        "fallback_instance_number"
    )


def test_fallback_volumes_are_refused_by_default(tmp_path):
    _write_series(tmp_path, odd_plane_rank=2)

    with pytest.raises(ValueError, match="fallback_instance_number"):
        ProcessOneDicomStudyToVolumesMappingBase(tmp_path)


@pytest.mark.parametrize("descending", [False, True])
def test_sub_volumes_are_ordered_geometrically(tmp_path, descending):
    _write_series(tmp_path, descending_instance_numbers=descending, volumes=3)

    series, volumes = _volumes(tmp_path)

    assert len(volumes) == 3
    assert series.get_slice_orderings() == {SliceOrdering.GEOMETRIC_SUBVOLUME}
    for index, volume in enumerate(volumes):
        _assert_in_position_order(volume)
        assert {v // 100 for v in _slice_values(volume)} == {index}
