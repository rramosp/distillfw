"""GCS and local filesystem storage abstraction for stateless task tracking."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse a `gs://bucket/path` URI into `(bucket_name, object_path)`."""
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Expected a valid gs:// URI, got: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


class StorageBackend:
    """Unified URI storage client supporting `gs://` and local emulation.

    If `DISTILLFW_LOCAL_GCS_ROOT` is set in the environment (or `local_root` is passed),
    `gs://<bucket>/<path>` URIs are transparently mapped to `<local_root>/<bucket>/<path>`.
    This guarantees 100% identical URI semantics across production GCP environments,
    local development, CI tests, and sample notebooks.
    """

    def __init__(self, project_id: str | None = None, local_root: str | Path | None = None) -> None:
        self.project_id = project_id
        env_local = os.environ.get("DISTILLFW_LOCAL_GCS_ROOT")
        self.local_root: Path | None = (
            Path(local_root)
            if local_root is not None
            else (Path(env_local) if env_local else None)
        )
        self._gcs_client = None

    @property
    def is_local_emulation(self) -> bool:
        return self.local_root is not None

    def _get_gcs_client(self):
        if self._gcs_client is None:
            from google.cloud import storage

            self._gcs_client = storage.Client(project=self.project_id)
        return self._gcs_client

    def _resolve_local_path(self, uri: str) -> Path:
        assert self.local_root is not None
        bucket, blob_path = parse_gcs_uri(uri)
        return self.local_root / bucket / blob_path

    def exists(self, uri: str) -> bool:
        """Check whether an object or directory exists at `uri`."""
        if self.is_local_emulation:
            return self._resolve_local_path(uri).exists()
        bucket_name, blob_path = parse_gcs_uri(uri)
        bucket = self._get_gcs_client().bucket(bucket_name)
        blob = bucket.blob(blob_path)
        if blob.exists():
            return True
        # Check if prefix has any blobs
        prefix = blob_path.rstrip("/") + "/"
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
        return len(blobs) > 0

    def read_text(self, uri: str, encoding: str = "utf-8") -> str:
        """Read UTF-8 text from `uri`."""
        if self.is_local_emulation:
            return self._resolve_local_path(uri).read_text(encoding=encoding)
        bucket_name, blob_path = parse_gcs_uri(uri)
        blob = self._get_gcs_client().bucket(bucket_name).blob(blob_path)
        return blob.download_as_text(encoding=encoding)

    def write_text(self, uri: str, content: str, encoding: str = "utf-8") -> None:
        """Atomically write text content to `uri`."""
        if self.is_local_emulation:
            target = self._resolve_local_path(uri)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding=encoding,
                dir=target.parent,
                delete=False,
            ) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)
            tmp_path.replace(target)
            return
        bucket_name, blob_path = parse_gcs_uri(uri)
        blob = self._get_gcs_client().bucket(bucket_name).blob(blob_path)
        blob.upload_from_string(content, content_type="text/plain; charset=utf-8")

    def read_json(self, uri: str) -> dict[str, Any]:
        """Read and deserialize a JSON object from `uri`."""
        return json.loads(self.read_text(uri))

    def write_json(self, uri: str, payload: dict[str, Any], indent: int = 2) -> None:
        """Serialize and atomically write a JSON object to `uri`."""
        self.write_text(uri, json.dumps(payload, indent=indent, sort_keys=False))

    def upload_file(self, local_path: str | Path, uri: str) -> None:
        """Upload a local file to `uri`."""
        src = Path(local_path)
        if self.is_local_emulation:
            dst = self._resolve_local_path(uri)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return
        bucket_name, blob_path = parse_gcs_uri(uri)
        blob = self._get_gcs_client().bucket(bucket_name).blob(blob_path)
        blob.upload_from_filename(str(src))

    def download_file(self, uri: str, local_path: str | Path) -> Path:
        """Download a single object from `uri` to `local_path`."""
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self.is_local_emulation:
            src = self._resolve_local_path(uri)
            shutil.copy2(src, dst)
            return dst
        bucket_name, blob_path = parse_gcs_uri(uri)
        blob = self._get_gcs_client().bucket(bucket_name).blob(blob_path)
        blob.download_to_filename(str(dst))
        return dst

    def upload_dir(self, local_dir: str | Path, uri_prefix: str) -> list[str]:
        """Recursively upload all files in `local_dir` to `uri_prefix`."""
        root = Path(local_dir)
        uploaded: list[str] = []
        base_uri = uri_prefix.rstrip("/")
        for file_path in sorted(root.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(root).as_posix()
                dest_uri = f"{base_uri}/{rel}"
                self.upload_file(file_path, dest_uri)
                uploaded.append(dest_uri)
        return uploaded

    def download_dir(self, uri_prefix: str, local_dir: str | Path) -> Path:
        """Recursively download all objects under `uri_prefix` to `local_dir`."""
        dst_root = Path(local_dir)
        dst_root.mkdir(parents=True, exist_ok=True)
        base_uri = uri_prefix.rstrip("/")
        for obj_uri in self.list_uris(base_uri):
            rel = obj_uri[len(base_uri) :].lstrip("/")
            self.download_file(obj_uri, dst_root / rel)
        return dst_root

    def list_uris(self, uri_prefix: str) -> list[str]:
        """List all object URIs under `uri_prefix`."""
        bucket_name, prefix = parse_gcs_uri(uri_prefix)
        clean_prefix = prefix.rstrip("/") + "/" if prefix else ""
        if self.is_local_emulation:
            root = self._resolve_local_path(f"gs://{bucket_name}/{clean_prefix}")
            if not root.exists():
                return []
            results: list[str] = []
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(self.local_root / bucket_name).as_posix()
                    results.append(f"gs://{bucket_name}/{rel}")
            return results
        bucket = self._get_gcs_client().bucket(bucket_name)
        return [
            f"gs://{bucket_name}/{blob.name}"
            for blob in bucket.list_blobs(prefix=clean_prefix)
            if not blob.name.endswith("/")
        ]
