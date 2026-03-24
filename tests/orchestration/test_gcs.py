import os
import pytest

from orchestration.gcs import (
    is_gcs_path,
    parse_gcs_path,
)


class TestIsGcsPath:
    def test_gcs_path(self):
        assert is_gcs_path("gs://bucket/path/to/file") is True

    def test_local_path(self):
        assert is_gcs_path("/local/path/to/file") is False

    def test_relative_path(self):
        assert is_gcs_path("relative/path") is False


class TestParseGcsPath:
    def test_simple_path(self):
        bucket, blob = parse_gcs_path("gs://my-bucket/path/to/file.csv")
        assert bucket == "my-bucket"
        assert blob == "path/to/file.csv"

    def test_bucket_only(self):
        bucket, blob = parse_gcs_path("gs://my-bucket/")
        assert bucket == "my-bucket"
        assert blob == ""

    def test_nested_path(self):
        bucket, blob = parse_gcs_path("gs://my-bucket/a/b/c/d.pt")
        assert bucket == "my-bucket"
        assert blob == "a/b/c/d.pt"
