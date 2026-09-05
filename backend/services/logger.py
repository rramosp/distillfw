"""Operations logger service for DistillFW."""

import time
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json


class OperationsLogger:
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.logs: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.listeners = []

    def log(self, message: str, level: str = "INFO", source: str = "SYSTEM", project_id: Optional[str] = None):
        entry = {
            "id": f"{int(time.time()*1000)}_{len(self.logs)}",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "level": level.upper(),
            "source": source,
            "project_id": project_id,
            "message": message,
        }
        with self.lock:
            self.logs.append(entry)
            if len(self.logs) > self.max_entries:
                self.logs.pop(0)

        # Print to stdout as well
        print(f"[{entry['timestamp']}] [{entry['level']}] [{entry['source']}] {message}")

        # Notify any active listeners
        for listener in list(self.listeners):
            try:
                listener(entry)
            except Exception:
                pass

    def get_logs(self, project_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        with self.lock:
            if project_id:
                filtered = [entry for entry in self.logs if entry.get("project_id") in (project_id, None)]
                return filtered[-limit:]
            return self.logs[-limit:]

    def clear(self):
        with self.lock:
            self.logs.clear()


operations_logger = OperationsLogger()
