"""
Tests for ExperimentLogger module.
Validates local CSV/JSON experiment recording, data integrity, and fallback behaviors.
"""
import json
import shutil
import tempfile
from pathlib import Path
import pytest
from src.evaluation.experiment_logger import ExperimentLogger, ExperimentRun


@pytest.fixture
def temp_log_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_experiment_logger_initialization(temp_log_dir):
    logger = ExperimentLogger(log_dir=temp_log_dir, use_wandb=False)
    assert logger.log_dir == temp_log_dir
    assert logger.json_path == temp_log_dir / "experiments.json"
    assert logger.csv_path == temp_log_dir / "experiments.csv"
    assert logger.get_runs() == []


def test_experiment_logger_log_run(temp_log_dir):
    logger = ExperimentLogger(log_dir=temp_log_dir, use_wandb=False)
    
    run = logger.log_run(
        model_name="test_yolov8n",
        checkpoint_path="models/best.pt",
        dataset_split="bdd100k_val_20",
        latency_ms=88.5,
        fps=11.3,
        config={"conf": 0.25, "imgsz": 640},
        metrics={"precision": 0.71, "recall": 0.60, "mAP50": 0.73},
        notes="Local test run verification",
        run_id="test_run_001",
    )
    
    assert isinstance(run, ExperimentRun)
    assert run.run_id == "test_run_001"
    assert run.latency_ms == 88.5
    assert run.fps == 11.3
    assert run.backend == "local_experiment_logger"
    
    # Check JSON file written
    assert logger.json_path.exists()
    runs = logger.get_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "test_run_001"
    assert runs[0]["metrics"]["mAP50"] == 0.73

    # Check CSV file written
    assert logger.csv_path.exists()
    with open(logger.csv_path, "r", encoding="utf-8") as f:
        csv_content = f.read()
    assert "test_run_001" in csv_content
    assert "0.73" in csv_content
    assert "local_experiment_logger" in csv_content


def test_experiment_logger_multiple_runs(temp_log_dir):
    logger = ExperimentLogger(log_dir=temp_log_dir, use_wandb=False)
    
    logger.log_run("model_a", "models/a.pt", "val", 50.0, 20.0, run_id="run_a")
    logger.log_run("model_b", "models/b.pt", "val", 45.0, 22.2, run_id="run_b")
    
    runs = logger.get_runs()
    assert len(runs) == 2
    assert runs[0]["run_id"] == "run_a"
    assert runs[1]["run_id"] == "run_b"


def test_experiment_logger_wandb_fallback(temp_log_dir):
    # Setting use_wandb=True should gracefully fallback if wandb not logged in/available
    logger = ExperimentLogger(log_dir=temp_log_dir, use_wandb=True)
    # Logging should still succeed without exceptions
    run = logger.log_run("model_fallback", "models/best.pt", "val", 60.0, 16.6)
    assert run.backend == "local_experiment_logger"
