"""Structured activity logging to console + timestamped local file with periodic & final GCS sync."""

from __future__ import annotations

import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distillfw.state import TaskWorkspace

_ACTIVE_LOGGER_LOCK = threading.RLock()
_ACTIVE_RUN_LOGGER: TaskRunLogger | None = None

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "distillfw") -> logging.Logger:
    """Return a child logger under the `distillfw` hierarchy."""
    if not name.startswith("distillfw"):
        name = f"distillfw.{name}"
    return logging.getLogger(name)


def get_active_run_logger() -> TaskRunLogger | None:
    """Return the currently active `TaskRunLogger`, if any."""
    with _ACTIVE_LOGGER_LOCK:
        return _ACTIVE_RUN_LOGGER


class TaskRunLogger:
    """Manages console + local timestamped file logging with periodic & final GCS uploads.

    - Creates a local log file prefixed with the start date & time:
      `<log_dir>/<YYYYMMDD_HHMMSS>_<command_name>.log`
    - Streams all `distillfw.*` logs to both stdout/stderr and the local file.
    - Runs a daemon background thread that uploads the local log file to
      `<task_uri>/logs/<filename>` on GCS every `sync_interval_seconds` (default: 60s).
    - Performs a final flush and upload to GCS when the command finishes.
    """

    def __init__(
        self,
        command_name: str,
        workspace: TaskWorkspace | None = None,
        log_dir: str | Path | None = None,
        sync_interval_seconds: float = 60.0,
    ) -> None:
        self.command_name = command_name
        self.workspace: TaskWorkspace | None = workspace
        self.sync_interval_seconds = sync_interval_seconds

        self.started_at = datetime.now()
        timestamp_prefix = self.started_at.strftime("%Y%m%d_%H%M%S")
        safe_cmd = command_name.replace(" ", "_").replace("/", "_")
        self.filename = f"{timestamp_prefix}_{safe_cmd}.log"

        resolved_dir = Path(
            log_dir
            if log_dir is not None
            else os.environ.get("DISTILLFW_LOG_DIR", "logs")
        )
        resolved_dir.mkdir(parents=True, exist_ok=True)
        self.local_log_path: Path = resolved_dir / self.filename

        self._stop_event = threading.Event()
        self._upload_lock = threading.Lock()
        self._sync_thread: threading.Thread | None = None
        self._file_handler: logging.FileHandler | None = None
        self._stream_handler: logging.StreamHandler | None = None
        self._is_owner = False
        self.upload_count: int = 0

    @property
    def gcs_log_uri(self) -> str | None:
        """Target GCS URI for this run's log file."""
        if self.workspace is None:
            return None
        return f"{self.workspace.logs_dir_uri}/{self.filename}"

    def attach_workspace(self, workspace: TaskWorkspace) -> None:
        """Attach or update the target `TaskWorkspace` once initialized."""
        with self._upload_lock:
            self.workspace = workspace

    def flush_and_upload(self) -> str | None:
        """Flush local file buffers and upload the current log file to GCS."""
        with self._upload_lock:
            if self._file_handler is not None:
                self._file_handler.flush()
            if self.workspace is None or not self.local_log_path.exists():
                return None
            dest_uri = f"{self.workspace.logs_dir_uri}/{self.filename}"
            try:
                self.workspace.storage.upload_file(self.local_log_path, dest_uri)
                self.upload_count += 1
                return dest_uri
            except Exception as exc:
                sys.stderr.write(f"[distillfw] Warning: failed to sync log to {dest_uri}: {exc}\n")
                return None

    def _background_sync_loop(self) -> None:
        while not self._stop_event.wait(timeout=self.sync_interval_seconds):
            self.flush_and_upload()

    def start(self) -> TaskRunLogger:
        """Attach log handlers and launch the background GCS sync thread."""
        global _ACTIVE_RUN_LOGGER
        with _ACTIVE_LOGGER_LOCK:
            if _ACTIVE_RUN_LOGGER is not None:
                # Reuse existing outer logger session
                if self.workspace is not None and _ACTIVE_RUN_LOGGER.workspace is None:
                    _ACTIVE_RUN_LOGGER.attach_workspace(self.workspace)
                return _ACTIVE_RUN_LOGGER
            _ACTIVE_RUN_LOGGER = self
            self._is_owner = True

        root_logger = logging.getLogger("distillfw")
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False

        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

        self._stream_handler = logging.StreamHandler(sys.stderr)
        self._stream_handler.setLevel(logging.INFO)
        self._stream_handler.setFormatter(formatter)

        self._file_handler = logging.FileHandler(self.local_log_path, mode="a", encoding="utf-8")
        self._file_handler.setLevel(logging.INFO)
        self._file_handler.setFormatter(formatter)

        root_logger.addHandler(self._stream_handler)
        root_logger.addHandler(self._file_handler)

        self._stop_event.clear()
        self._sync_thread = threading.Thread(
            target=self._background_sync_loop,
            name=f"distillfw-log-sync-{self.filename}",
            daemon=True,
        )
        self._sync_thread.start()

        root_logger.info(
            "Started command '%s' | local log: %s%s",
            self.command_name,
            self.local_log_path,
            f" | GCS log: {self.gcs_log_uri}" if self.gcs_log_uri else "",
        )
        return self

    def close(self) -> str | None:
        """Stop background thread, log completion, perform final GCS upload, and detach handlers."""
        global _ACTIVE_RUN_LOGGER
        if not self._is_owner:
            return None

        root_logger = logging.getLogger("distillfw")
        elapsed = (datetime.now() - self.started_at).total_seconds()
        root_logger.info(
            "Finished command '%s' in %.2fs | syncing final log to %s",
            self.command_name,
            elapsed,
            self.gcs_log_uri or self.local_log_path,
        )

        self._stop_event.set()
        if self._sync_thread is not None and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        final_uri = self.flush_and_upload()

        if self._file_handler is not None:
            root_logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None
        if self._stream_handler is not None:
            root_logger.removeHandler(self._stream_handler)
            self._stream_handler = None

        with _ACTIVE_LOGGER_LOCK:
            if _ACTIVE_RUN_LOGGER is self:
                _ACTIVE_RUN_LOGGER = None
        self._is_owner = False
        return final_uri

    def __enter__(self) -> TaskRunLogger:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_val is not None:
            logging.getLogger("distillfw").error(
                "Command '%s' terminated with %s: %s",
                self.command_name,
                exc_type.__name__ if exc_type else "Error",
                exc_val,
            )
        self.close()
