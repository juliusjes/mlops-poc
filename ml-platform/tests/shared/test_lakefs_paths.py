from ml_platform.shared.services.lakefs_service import LakeFSService


class FakeLakeFSFileSystem:
    def __init__(self, result=None, error=None):
        self.result = [] if result is None else result
        self.error = error
        self.locations = []

    def ls(self, location):
        self.locations.append(location)
        if self.error:
            raise self.error
        return self.result


def service_with_filesystem(filesystem):
    """Avoid service initialization and all external configuration in unit tests."""
    service = object.__new__(LakeFSService)
    service.fs = filesystem
    return service


def test_is_path_lakefs_accepts_existing_configured_path():
    filesystem = FakeLakeFSFileSystem(result=[{"name": "part-00000.parquet"}])
    service = service_with_filesystem(filesystem)

    assert service.is_path_lakefs("pytest1/main/rawdata/") is True
    assert filesystem.locations == ["pytest1/main/rawdata/"]


def test_is_path_lakefs_requires_repository_and_branch_before_lookup():
    filesystem = FakeLakeFSFileSystem(result=[{"name": "unused"}])
    service = service_with_filesystem(filesystem)

    invalid_locations = [
        None,
        "",
        "repo-only",
        "/repo/main/data",
        "lakefs://repo/main/data",
        "repo//main/data",
        "repo/../data",
        "repo/./data",
    ]

    for location in invalid_locations:
        assert service.is_path_lakefs(location) is False

    assert filesystem.locations == []


def test_is_path_lakefs_returns_false_when_object_path_is_missing():
    filesystem = FakeLakeFSFileSystem(error=FileNotFoundError())
    service = service_with_filesystem(filesystem)

    assert service.is_path_lakefs("pytest1/main/missing/") is False
    assert filesystem.locations == ["pytest1/main/missing/"]


def test_is_path_lakefs_returns_false_for_an_empty_listing():
    filesystem = FakeLakeFSFileSystem(result=[])
    service = service_with_filesystem(filesystem)

    assert service.is_path_lakefs("pytest1/main/empty/") is False
