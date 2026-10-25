import json
from pathlib import Path
import pytest
from src.config import CONFIG
from src.predictor import predict_video


def test_baseline_json_exists_and_valid():
    """ตรวจสอบว่าไฟล์ baseline.json มีโครงสร้างครบถ้วน"""
    baseline_path = CONFIG["CANARY_DIR"] / "baseline.json"
    assert baseline_path.exists(), "baseline.json ต้องอยู่ใน assets/canary/"

    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "threshold" in data
    assert "class_index_fake" in data
    assert data["class_index_fake"] == 1
    assert "canary_targets" in data


def test_canary_evaluation_pipeline():
    """จำลองการรัน Canary self-test กับ target baseline"""
    baseline_path = CONFIG["CANARY_DIR"] / "baseline.json"
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    targets = data.get("canary_targets", {})
    assert len(targets) >= 2
    assert "fake_sample.mp4" in targets
    assert "real_sample.mp4" in targets
