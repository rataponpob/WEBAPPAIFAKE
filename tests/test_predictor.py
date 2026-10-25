import numpy as np
import pytest
from src.predictor import PredictionResult, predict_video


class MockSigmoidModel:
    """Mock model ที่จำลองการส่งออกค่า Sigmoid P(index 1 = REAL)"""

    def __init__(self, output_value: float):
        self.output_value = output_value

    def predict(self, batch, verbose=0):
        # batch shape: (B, H, W, C)
        n = len(batch)
        return np.full((n, 1), self.output_value, dtype=np.float32)


def test_label_mapping_real_prediction():
    """เมื่อโมเดลส่งค่า Sigmoid ต่ำ (~0.05) แปลว่าโมเดลทายคลาส 0 (REAL)
    ระบบต้องแปลงเป็น prob(FAKE) = 0.05 และตัดสินเป็น 'real'
    """
    mock_model = MockSigmoidModel(output_value=0.05)
    dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]

    res = predict_video(
        model=mock_model,
        frames=dummy_frames,
        threshold=0.55,
        class_index_fake=1,
        preprocessor=lambda x: x,
    )

    assert isinstance(res, PredictionResult)
    assert res.label == "real"
    assert pytest.approx(res.mean_prob_fake, 0.01) == 0.05
    assert res.borderline is False


def test_label_mapping_fake_prediction():
    """เมื่อโมเดลส่งค่า Sigmoid สูง (~0.95) แปลว่าโมเดลทายคลาส 1 (FAKE)
    ระบบต้องแปลงเป็น prob(FAKE) = 0.95 และตัดสินเป็น 'fake'
    """
    mock_model = MockSigmoidModel(output_value=0.95)
    dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]

    res = predict_video(
        model=mock_model,
        frames=dummy_frames,
        threshold=0.55,
        class_index_fake=1,
        preprocessor=lambda x: x,
    )

    assert res.label == "fake"
    assert pytest.approx(res.mean_prob_fake, 0.01) == 0.95
    assert res.borderline is False


def test_borderline_band_detection():
    """ทดสอบกรณีค่าความน่าจะเป็นตกอยู่ในช่วง Borderline (0.45 - 0.65)"""
    mock_model = MockSigmoidModel(output_value=0.52)  # prob(fake) = 0.52
    dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(8)]

    res = predict_video(
        model=mock_model,
        frames=dummy_frames,
        threshold=0.55,
        class_index_fake=1,
        borderline_band=(0.45, 0.65),
        preprocessor=lambda x: x,
    )

    assert pytest.approx(res.mean_prob_fake, 0.01) == 0.52
    assert res.borderline is True
    assert "ไม่แน่นอน" in res.confidence_level


def test_empty_frames_raises_error():
    """ทดสอบกรณีส่ง frames ว่างเปล่า ต้องโยน ValueError"""
    mock_model = MockSigmoidModel(output_value=0.5)
    with pytest.raises(ValueError):
        predict_video(model=mock_model, frames=[], threshold=0.55)
