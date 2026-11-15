import os
import pickle
from pathlib import Path

import pytest

from dcm_classifier import image_type_inference
from dcm_classifier.image_type_inference import (
    ImageTypeClassifierBase,
    allowed_cpu_count,
    cgroup_cpu_quota,
    cpu_limit_variables,
)
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase

inference_model_path: Path = (
    Path(__file__).parents[2] / "models" / "ova_rf_classifier.onnx"
)
dwi_directory: Path = (
    Path(__file__).parents[1]
    / "testing_data"
    / "anonymized_testing_data"
    / "anonymized_dwi_series"
)


def test_session_uses_requested_thread_count():
    inferer = ImageTypeClassifierBase(
        classification_model_filename=inference_model_path, num_threads=2
    )
    options = inferer.get_inference_session().get_session_options()
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1


@pytest.fixture
def no_cpu_limits(monkeypatch):
    """Hide every CPU limit of the host so each test sets only its own."""
    for variable in cpu_limit_variables:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(image_type_inference, "cgroup_cpu_quota", lambda: None)


def write_cgroup(directory: Path, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content)


@pytest.mark.usefixtures("no_cpu_limits")
def test_no_limit_leaves_thread_count_to_onnxruntime():
    assert allowed_cpu_count() is None
    inferer = ImageTypeClassifierBase(
        classification_model_filename=inference_model_path
    )
    assert inferer.num_threads is None
    options = inferer.get_inference_session().get_session_options()
    assert options.intra_op_num_threads == 0


@pytest.mark.usefixtures("no_cpu_limits")
def test_cpu_affinity_limits_thread_count(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)
    assert allowed_cpu_count() == 3
    inferer = ImageTypeClassifierBase(
        classification_model_filename=inference_model_path
    )
    options = inferer.get_inference_session().get_session_options()
    assert options.intra_op_num_threads == 3


@pytest.mark.parametrize("variable", cpu_limit_variables)
@pytest.mark.usefixtures("no_cpu_limits")
def test_environment_variable_limits_thread_count(monkeypatch, variable):
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(range(48)), raising=False
    )
    monkeypatch.setenv(variable, "4")
    assert allowed_cpu_count() == 4


@pytest.mark.parametrize("value", ["", "four", "0", "-2"])
@pytest.mark.usefixtures("no_cpu_limits")
def test_unusable_environment_variable_is_ignored(monkeypatch, value):
    monkeypatch.setenv("NSLOTS", value)
    assert allowed_cpu_count() is None


@pytest.mark.usefixtures("no_cpu_limits")
def test_nested_omp_num_threads_uses_outer_level(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4,2")
    assert allowed_cpu_count() == 4


@pytest.mark.usefixtures("no_cpu_limits")
def test_smallest_limit_wins(monkeypatch):
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(range(8)), raising=False
    )
    monkeypatch.setattr(image_type_inference, "cgroup_cpu_quota", lambda: 1.5)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    assert allowed_cpu_count() == 2


@pytest.mark.usefixtures("no_cpu_limits")
def test_fractional_cgroup_quota_allows_one_thread(monkeypatch):
    monkeypatch.setattr(image_type_inference, "cgroup_cpu_quota", lambda: 0.5)
    assert allowed_cpu_count() == 1


def test_cgroup_v2_quota_inside_container(tmp_path):
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text("0::/\n")
    root = tmp_path / "sys"
    write_cgroup(root, {"cpu.max": "200000 100000\n"})
    assert cgroup_cpu_quota(proc_cgroup, root) == 2.0


def test_cgroup_v2_unlimited(tmp_path):
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text("0::/\n")
    root = tmp_path / "sys"
    write_cgroup(root, {"cpu.max": "max 100000\n"})
    assert cgroup_cpu_quota(proc_cgroup, root) is None


def test_cgroup_v2_quota_on_ancestor(tmp_path):
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text("0::/system.slice/job_17/step_0\n")
    root = tmp_path / "sys"
    write_cgroup(root / "system.slice" / "job_17", {"cpu.max": "400000 100000\n"})
    write_cgroup(
        root / "system.slice" / "job_17" / "step_0", {"cpu.max": "max 100000\n"}
    )
    assert cgroup_cpu_quota(proc_cgroup, root) == 4.0


def test_cgroup_v1_quota(tmp_path):
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text(
        "12:memory:/docker/abc\n5:cpu,cpuacct:/docker/abc\n1:name=systemd:/docker/abc\n"
    )
    root = tmp_path / "sys"
    write_cgroup(
        root / "cpu,cpuacct",
        {"cpu.cfs_quota_us": "150000\n", "cpu.cfs_period_us": "100000\n"},
    )
    assert cgroup_cpu_quota(proc_cgroup, root) == 1.5


def test_cgroup_v1_unlimited(tmp_path):
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text("5:cpu,cpuacct:/\n")
    root = tmp_path / "sys"
    write_cgroup(
        root / "cpu,cpuacct",
        {"cpu.cfs_quota_us": "-1\n", "cpu.cfs_period_us": "100000\n"},
    )
    assert cgroup_cpu_quota(proc_cgroup, root) is None


def test_no_cgroup_filesystem(tmp_path):
    assert cgroup_cpu_quota(tmp_path / "missing", tmp_path) is None


@pytest.mark.parametrize("num_threads", [0, -1])
def test_non_positive_thread_count_is_rejected(num_threads):
    with pytest.raises(ValueError, match="num_threads must be at least 1"):
        ImageTypeClassifierBase(
            classification_model_filename=inference_model_path,
            num_threads=num_threads,
        )


@pytest.mark.filterwarnings(
    "ignore:SeriesNumber 2 .* Siemens mosaic:UserWarning",
    "ignore:Missing features for SeriesNumber 2:UserWarning",
)
def test_session_is_reused_across_inference_calls():
    inferer = ImageTypeClassifierBase(
        classification_model_filename=inference_model_path, num_threads=1
    )
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=dwi_directory, inferer=inferer
    )
    study.run_inference()
    session = inferer.get_inference_session()
    study.run_inference()
    assert inferer.get_inference_session() is session


def test_session_is_rebuilt_when_model_filename_changes():
    inferer = ImageTypeClassifierBase(
        classification_model_filename=inference_model_path, num_threads=1
    )
    session = inferer.get_inference_session()
    inferer.classification_model_filename = (
        inference_model_path.parent / "rf_classifier.onnx"
    )
    assert inferer.get_inference_session() is not session


def test_classifier_with_open_session_can_be_pickled():
    inferer = ImageTypeClassifierBase(
        classification_model_filename=inference_model_path, num_threads=1
    )
    inferer.get_inference_session()
    restored = pickle.loads(pickle.dumps(inferer))
    assert restored.num_threads == 1
    assert restored.get_inference_session() is not None


@pytest.mark.parametrize(
    ("row_x", "column_z", "plane"),
    [
        (1.0, 0.0, "ax"),
        (0.0, 1.0, "sag"),
        (1.0, 1.0, "cor"),
    ],
)
def test_infer_acquisition_plane(row_x, column_z, plane):
    features = {
        "ImageOrientationPatient_0": row_x,
        "ImageOrientationPatient_5": column_z,
    }
    assert ImageTypeClassifierBase().infer_acquisition_plane(features) == plane


def test_unknown_isotropy_is_reported_as_not_isotropic(get_data_dir, monkeypatch):
    inferer = ImageTypeClassifierBase(
        classification_model_filename=inference_model_path, num_threads=1
    )
    monkeypatch.setattr(inferer, "infer_isotropic", lambda **_: None)
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=get_data_dir, inferer=inferer
    )
    study.run_inference()
    for series in study.series_dictionary.values():
        assert series.get_is_isotropic() is False
        for volume in series.get_volume_list():
            assert volume.get_is_isotropic() is False


def test_run_inference_without_a_series_raises():
    inferer = ImageTypeClassifierBase(
        classification_model_filename=inference_model_path
    )
    with pytest.raises(AttributeError, match="No series is set"):
        inferer.run_inference()
