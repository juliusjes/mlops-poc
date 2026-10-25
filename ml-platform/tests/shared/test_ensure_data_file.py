from pathlib import Path

from ml_platform.shared.data_processing.data_processing import ensure_data_file


def test_ensure_data_file_uses_canonical_argument_when_pipeline_passes_none(
    tmp_path, monkeypatch
):
    data_file = tmp_path / "data" / "LSWMD.pkl"
    download_file = tmp_path / "download_wafermap.sh"
    download_file.touch()
    calls = []

    def fake_run(command, check):
        calls.append((command, check))
        data_file.parent.mkdir(exist_ok=True)
        data_file.write_bytes(b"fixture")

    monkeypatch.setattr(
        "ml_platform.shared.data_processing.data_processing.DATA_FILE_PATH",
        data_file,
    )
    monkeypatch.setattr(
        "ml_platform.shared.data_processing.data_processing.subprocess.run",
        fake_run,
    )

    result = ensure_data_file(data_file=None, download_file=download_file)

    assert result == data_file
    assert calls == [(["bash", str(download_file), str(data_file.parent)], True)]


def test_ensure_data_file_does_not_download_when_file_exists(tmp_path, monkeypatch):
    data_file = tmp_path / "LSWMD.pkl"
    data_file.write_bytes(b"fixture")
    download_file = tmp_path / "download_wafermap.sh"
    download_file.touch()

    monkeypatch.setattr(
        "ml_platform.shared.data_processing.data_processing.subprocess.run",
        lambda *_: (_ for _ in ()).throw(AssertionError("download should not run")),
    )

    assert ensure_data_file(data_file=data_file, download_file=download_file) == data_file
