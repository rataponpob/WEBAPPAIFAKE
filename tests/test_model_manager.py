from pathlib import Path
import pytest
from src.model_loader import (
    get_available_models,
    get_active_model_path,
    set_active_model_path,
    is_model_available,
)
from src.config import PROJECT_ROOT


def test_get_available_models():
    models = get_available_models()
    assert isinstance(models, list)
    # ควรพบ ResNet50_4dataset.keras
    names = [m.name for m in models]
    assert "ResNet50_4dataset.keras" in names


def test_active_model_path():
    path = get_active_model_path()
    assert isinstance(path, Path)
    assert path.is_file()

    # ทดสอบ set_active_model_path
    set_active_model_path(path)
    assert get_active_model_path() == path


def test_is_model_available():
    active = get_active_model_path()
    assert is_model_available(active) is True
    assert is_model_available("non_existent_model.keras") is False
