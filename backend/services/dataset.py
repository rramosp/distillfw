"""Dataset ingestion, validation, and splitting service."""

import json
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

from backend.services.storage import storage_service
from backend.services.logger import operations_logger


class DatasetService:
    def validate_lines(self, lines: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        valid_rows = []
        errors = []
        permitted_splits = {"train", "val", "test"}

        for idx, line in enumerate(lines, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
            except json.JSONDecodeError as e:
                errors.append(f"Line {idx}: Invalid JSON syntax - {e}")
                continue

            if not isinstance(item, dict):
                errors.append(f"Line {idx}: Row must be a JSON object")
                continue

            if "prompt" not in item or not str(item["prompt"]).strip():
                errors.append(f"Line {idx}: Missing or empty 'prompt' field")
                continue

            split_val = item.get("split")
            if split_val is not None:
                if split_val not in permitted_splits:
                    errors.append(f"Line {idx}: Invalid split '{split_val}'. Allowed: 'train', 'val', 'test'")
                    continue

            valid_rows.append(item)

        return valid_rows, errors

    def split_dataset(
        self,
        rows: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        # Check if all rows already have valid splits
        has_existing_splits = all("split" in row and row["split"] in {"train", "val", "test"} for row in rows)
        if has_existing_splits and len(rows) > 0:
            return rows

        # Otherwise perform random split with configurable seed
        rng = random.Random(random_seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Remainder goes to test split
        n_test = n - n_train - n_val

        split_rows = []
        for i, row in enumerate(shuffled):
            item = dict(row)
            if i < n_train:
                item["split"] = "train"
            elif i < n_train + n_val:
                item["split"] = "val"
            else:
                item["split"] = "test"
            split_rows.append(item)

        return split_rows

    def ingest_and_split(
        self,
        bucket_name: str,
        project_id: str,
        raw_jsonl_content: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc).isoformat()
        operations_logger.log(
            f"Starting dataset ingestion and splitting for project '{project_id}'",
            level="INFO",
            source="DATASET",
            project_id=project_id
        )

        # 1. Save raw dataset as data/input_dataset.jsonl
        input_path = f"{project_id}/data/input_dataset.jsonl"
        storage_service.write_file(bucket_name, input_path, raw_jsonl_content)

        # 2. Validate rows
        lines = raw_jsonl_content.splitlines()
        valid_rows, errors = self.validate_lines(lines)

        if not valid_rows:
            msg = f"Dataset validation failed with 0 valid rows: {errors[:5]}"
            operations_logger.log(msg, level="ERROR", source="DATASET", project_id=project_id)
            storage_service.record_history(
                bucket_name, project_id, "DATASET_INGESTION", "FAILED",
                {"total_lines": len(lines), "errors": errors},
                msg, start_time
            )
            return {"success": False, "errors": errors, "total_lines": len(lines)}

        # 3. Apply splitting logic
        split_rows = self.split_dataset(valid_rows, train_ratio, val_ratio, test_ratio, random_seed)

        # 4. Save data/split_dataset.jsonl
        split_content = "\n".join(json.dumps(row) for row in split_rows) + "\n"
        storage_service.write_file(bucket_name, f"{project_id}/data/split_dataset.jsonl", split_content)

        counts = {
            "total": len(split_rows),
            "train": sum(1 for r in split_rows if r.get("split") == "train"),
            "val": sum(1 for r in split_rows if r.get("split") == "val"),
            "test": sum(1 for r in split_rows if r.get("split") == "test"),
        }

        operations_logger.log(
            f"Dataset ready: {counts['total']} rows (Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']})",
            level="SUCCESS",
            source="DATASET",
            project_id=project_id
        )

        storage_service.record_history(
            bucket_name, project_id, "DATASET_SPLIT", "SUCCESS",
            {
                "counts": counts,
                "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
                "random_seed": random_seed,
                "warnings": errors
            },
            f"Created split_dataset.jsonl with {counts['total']} samples.",
            start_time
        )

        return {
            "success": True,
            "counts": counts,
            "warnings": errors,
            "sample_rows": split_rows[:5]
        }

    def get_summary(self, bucket_name: str, project_id: str) -> Dict[str, Any]:
        split_path = f"{project_id}/data/split_dataset.jsonl"
        if not storage_service.file_exists(bucket_name, split_path):
            input_path = f"{project_id}/data/input_dataset.jsonl"
            if storage_service.file_exists(bucket_name, input_path):
                raw = storage_service.read_file(bucket_name, input_path)
                lines = [l for l in raw.splitlines() if l.strip()]
                return {"has_dataset": True, "is_split": False, "total_lines": len(lines)}
            return {"has_dataset": False, "is_split": False}

        raw = storage_service.read_file(bucket_name, split_path)
        lines = [l for l in raw.splitlines() if l.strip()]
        rows = [json.loads(l) for l in lines]
        counts = {
            "total": len(rows),
            "train": sum(1 for r in rows if r.get("split") == "train"),
            "val": sum(1 for r in rows if r.get("split") == "val"),
            "test": sum(1 for r in rows if r.get("split") == "test"),
        }
        return {
            "has_dataset": True,
            "is_split": True,
            "counts": counts,
            "samples": rows[:5]
        }

    def clear(self, bucket_name: str, project_id: str) -> Dict[str, Any]:
        storage_service.delete_file(bucket_name, f"{project_id}/data/split_dataset.jsonl")
        storage_service.delete_file(bucket_name, f"{project_id}/data/input_dataset.jsonl")
        operations_logger.log(f"Cleared dataset for project '{project_id}'", level="INFO", source="DATASET", project_id=project_id)
        return {"status": "CLEARED", "project_id": project_id}


dataset_service = DatasetService()

