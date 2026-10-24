from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np


def extract_frames(
    video_path: Union[str, Path],
    n_frames: int = 16,
    img_size: Tuple[int, int] = (224, 224),
) -> List[np.ndarray]:
    """Uniform sampling ตามสูตร idx = int(total * f / (n+1)), f = 1..n
    คืน list ของ RGB ndarray ขนาด img_size (เรียงตามลำดับเวลา)
    เหมือนในโน้ตบุ๊ก cell 46 เป๊ะ เพื่อผลลัพธ์ที่ reproducible
    """
    video_str = str(video_path)
    cap = cv2.VideoCapture(video_str)
    frames: List[np.ndarray] = []

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return frames

        idxs = [int(total * f / (n_frames + 1)) for f in range(1, n_frames + 1)]
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            # แปลง BGR เป็น RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resize เป็นขนาด input โมเดล
            frame_resized = cv2.resize(frame_rgb, img_size)
            frames.append(frame_resized)
    finally:
        cap.release()

    return frames


def extract_frames_dual(
    video_path: Union[str, Path],
    n_frames: int = 16,
    img_size: Tuple[int, int] = (224, 224),
    max_dim: int = 720,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """สุ่มดึงเฟรม 2 ชุดพร้อมกัน:
    1. model_frames: ขนาด (224, 224) สำหรับส่งเข้าโมเดล ResNet50
    2. display_frames: คงอัตราส่วนจริง (Aspect Ratio) เดิมของคลิปไว้ ไม่บีบภาพแบนเป็นสี่เหลี่ยม
    """
    video_str = str(video_path)
    cap = cv2.VideoCapture(video_str)
    model_frames: List[np.ndarray] = []
    display_frames: List[np.ndarray] = []

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return model_frames, display_frames

        idxs = [int(total * f / (n_frames + 1)) for f in range(1, n_frames + 1)]
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            # คำนวณขนาด display_frame โดยรักษาอัตราส่วนเดิมไว้ 100%
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                nw = int(w * scale)
                nh = int(h * scale)
                nw = nw if nw % 2 == 0 else nw - 1
                nh = nh if nh % 2 == 0 else nh - 1
                disp = cv2.resize(frame_rgb, (nw, nh))
            else:
                nw = w if w % 2 == 0 else w - 1
                nh = h if h % 2 == 0 else h - 1
                if nw != w or nh != h:
                    disp = cv2.resize(frame_rgb, (nw, nh))
                else:
                    disp = frame_rgb

            display_frames.append(disp)

            # สำหรับโมเดล ResNet50
            frame_resized = cv2.resize(frame_rgb, img_size)
            model_frames.append(frame_resized)
    finally:
        cap.release()

    return model_frames, display_frames
