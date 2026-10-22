"""
Local Experiment Logger Module.
Provides lightweight, reproducible local CSV and JSON experiment tracking
as an alternative to cloud logging platforms (such as Weights & Biases).
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    """Represents a single experiment or benchmark evaluation run."""
    run_id: str
    timestamp: str
    model_name: str
    checkpoint_path: str
    dataset_split: str
    config: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    fps: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    backend: str = "local_experiment_logger"


class ExperimentLogger:
    """
    Lightweight local experiment logger storing runs in both JSON and CSV formats.
    
    Ensures complete scientific reproducibility on edge/laptop environments
    without requiring third-party cloud accounts, internet connectivity, or credentials.
    Optional Weights & Biases (wandb) hook is supported if installed.
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "results",
        json_filename: str = "experiments.json",
        csv_filename: str = "experiments.csv",
        use_wandb: bool = False,
        wandb_project: str = "auto_drive_perception",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.log_dir / json_filename
        self.csv_path = self.log_dir / csv_filename
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self._wandb_run = None

        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Attempts to initialize wandb if available; falls back gracefully."""
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                mode="offline",  # Default to offline mode for safety
            )
            logger.info("Optional Weights & Biases (wandb) initialized in offline mode.")
        except ImportError:
            logger.warning("wandb package not installed. Continuing with local logging only.")
            self.use_wandb = False
        except Exception as e:
            logger.warning(f"wandb initialization failed ({e}). Continuing with local logging.")
            self.use_wandb = False

    def log_run(
        self,
        model_name: str,
        checkpoint_path: Union[str, Path],
        dataset_split: str,
        latency_ms: float,
        fps: float,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
        run_id: Optional[str] = None,
    ) -> ExperimentRun:
        """
        Log a complete experiment run to local JSON and CSV stores.

        Args:
            model_name: Human-readable model identifier (e.g. 'yolov8n_baseline').
            checkpoint_path: Path to model weights file.
            dataset_split: Dataset split evaluated (e.g. 'bdd100k_val_20').
            latency_ms: Mean or median execution latency in milliseconds.
            fps: Effective throughput in frames per second.
            config: Optional configuration dictionary.
            metrics: Optional evaluation metrics dictionary (e.g. precision, recall, mAP).
            notes: Optional explanatory notes.
            run_id: Optional custom run identifier; auto-generated if omitted.

        Returns:
            The recorded ExperimentRun instance.
        """
        now = datetime.now(timezone.utc)
        iso_timestamp = now.isoformat()
        if run_id is None:
            time_tag = now.strftime("%Y%m%d_%H%M%S")
            safe_model = model_name.replace(" ", "_").replace("/", "_")
            run_id = f"exp_{safe_model}_{time_tag}"

        run = ExperimentRun(
            run_id=run_id,
            timestamp=iso_timestamp,
            model_name=model_name,
            checkpoint_path=str(checkpoint_path),
            dataset_split=dataset_split,
            config=config or {},
            latency_ms=round(float(latency_ms), 3),
            fps=round(float(fps), 2),
            metrics=metrics or {},
            notes=notes,
            backend="local_experiment_logger",
        )

        # 1. Append to JSON store
        self._append_json(run)

        # 2. Append to CSV store
        self._append_csv(run)

        # 3. Optional wandb log
        if self.use_wandb and self._wandb_run is not None:
            try:
                import wandb
                wandb.log({
                    "latency_ms": run.latency_ms,
                    "fps": run.fps,
                    **{f"metrics/{k}": v for k, v in run.metrics.items() if isinstance(v, (int, float))},
                })
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

        logger.info(f"Experiment run logged: {run.run_id} (latency: {run.latency_ms}ms, fps: {run.fps})")
        return run

    def _append_json(self, run: ExperimentRun) -> None:
        """Appends run record to JSON list file."""
        runs: List[Dict[str, Any]] = []
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        runs = json.loads(content)
            except Exception as e:
                logger.warning(f"Could not parse existing JSON log ({e}); creating new list.")
                runs = []

        runs.append(asdict(run))

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)

    def _append_csv(self, run: ExperimentRun) -> None:
        """Appends run record as a flattened row in CSV file."""
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0

        fieldnames = [
            "run_id",
            "timestamp",
            "model_name",
            "checkpoint_path",
            "dataset_split",
            "latency_ms",
            "fps",
            "precision",
            "recall",
            "mAP50",
            "mAP50_95",
            "notes",
            "backend",
        ]

        row = {
            "run_id": run.run_id,
            "timestamp": run.timestamp,
            "model_name": run.model_name,
            "checkpoint_path": run.checkpoint_path,
            "dataset_split": run.dataset_split,
            "latency_ms": run.latency_ms,
            "fps": run.fps,
            "precision": run.metrics.get("precision", run.metrics.get("P", "")),
            "recall": run.metrics.get("recall", run.metrics.get("R", "")),
            "mAP50": run.metrics.get("mAP50", run.metrics.get("mAP_0.5", "")),
            "mAP50_95": run.metrics.get("mAP50-95", run.metrics.get("mAP_0.5:0.95", "")),
            "notes": run.notes,
            "backend": run.backend,
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def get_runs(self) -> List[Dict[str, Any]]:
        """Reads all logged experiment runs from the JSON store."""
        if not self.json_path.exists():
            return []
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read runs from {self.json_path}: {e}")
            return []
