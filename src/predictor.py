from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional
import numpy as np


@dataclass
class PredictionResult:
    """ผลการทำนายระดับวิดีโอ"""

    label: str  # "fake" หรือ "real"
    mean_prob_fake: float  # ความน่าจะเป็น fake เฉลี่ย (0.0 - 1.0)
    frame_probs: List[float] = field(
        default_factory=list
    )  # ค่า prob(fake) รายเฟรม
    n_frames_used: int = 0  # จำนวนเฟรมที่ใช้จริง
    threshold: float = 0.55  # เกณฑ์ตัดสินที่ใช้
    borderline: bool = False  # อยู่ในช่วงที่ไม่แน่นอนหรือไม่ (เช่น 0.45 - 0.65)
    confidence_level: str = "สูง"  # "สูง", "ปานกลาง", หรือ "ไม่แน่นอน"
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "mean_prob_fake": round(self.mean_prob_fake, 4),
            "frame_probs": [round(p, 4) for p in self.frame_probs],
            "n_frames_used": self.n_frames_used,
            "threshold": self.threshold,
            "borderline": self.borderline,
            "confidence_level": self.confidence_level,
        }


def get_default_preprocessor() -> Callable[[np.ndarray], np.ndarray]:
    """โหลด resnet50 preprocess_input จาก tensorflow"""
    try:
        from tensorflow.keras.applications.resnet50 import preprocess_input

        return preprocess_input
    except ImportError:
        # Fallback กรณี mock ใน unit test ที่ไม่มี tf
        def dummy_preprocess(x: np.ndarray) -> np.ndarray:
            return x

        return dummy_preprocess


def predict_video(
    model: Any,
    frames: List[np.ndarray],
    threshold: float = 0.55,
    class_index_fake: int = 1,
    borderline_band: tuple = (0.45, 0.65),
    preprocessor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> PredictionResult:
    """ทำนายวิดีโอจาก list ของ RGB frames

    ตรรกะแปลง label (จุดเดียวในทั้งระบบ):
    - ถ้า class_index_fake = 1: prob(fake) = raw_sigmoid (index 1 = FAKE)
    - ถ้า class_index_fake = 0: prob(fake) = 1.0 - raw_sigmoid (index 0 = FAKE)
    """
    if not frames:
        raise ValueError("frames ว่างเปล่า — ไม่สามารถทำนายได้")

    if preprocessor is None:
        preprocessor = get_default_preprocessor()

    batch = np.array(frames, dtype="float32")
    batch = preprocessor(batch)

    # predict แบบ batch เดียวครบทุกเฟรม
    preds = model.predict(batch, verbose=0)
    preds = np.asarray(preds)

    if preds.ndim == 1:
        raw = preds
        frame_probs = raw if class_index_fake == 1 else (1.0 - raw)
    elif preds.shape[-1] == 1:
        raw = preds[:, 0]
        # จุดสำคัญ: ถ้า sigmoid output = P(REAL) และ FAKE=0 -> prob(FAKE) = 1.0 - raw
        frame_probs = raw if class_index_fake == 1 else (1.0 - raw)
    else:
        frame_probs = preds[:, class_index_fake]

    frame_probs_list = [float(p) for p in frame_probs]
    mean_prob = float(np.mean(frame_probs_list))

    label = "fake" if mean_prob >= threshold else "real"

    low_b, high_b = borderline_band
    is_borderline = low_b <= mean_prob <= high_b

    # คำนวณระดับความมั่นใจ
    dist_from_threshold = abs(mean_prob - threshold)
    if is_borderline:
        confidence = "ไม่แน่นอน (Borderline)"
    elif dist_from_threshold >= 0.25:
        confidence = "สูง (High)"
    else:
        confidence = "ปานกลาง (Medium)"

    return PredictionResult(
        label=label,
        mean_prob_fake=mean_prob,
        frame_probs=frame_probs_list,
        n_frames_used=len(frames),
        threshold=threshold,
        borderline=is_borderline,
        confidence_level=confidence,
    )
