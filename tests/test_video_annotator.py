import numpy as np
import pytest
from pathlib import Path
from src.video_annotator import draw_frame_overlay, create_sampled_clip


def test_draw_frame_overlay_shape_and_type():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = draw_frame_overlay(
        frame_rgb=frame,
        frame_idx=5,
        total_frames=100,
        prob_fake=0.85,
        threshold=0.55,
        fps=30.0,
    )
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    # ตรวจสอบว่ามีการวาดสีลงในภาพ (ไม่เป็น 0 ทั้งหมด)
    assert np.any(annotated > 0)


def test_create_sampled_clip(tmp_path):
    frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
    probs = [0.1, 0.9, 0.4, 0.8]
    out_video = tmp_path / "test_out.mp4"

    res_path = create_sampled_clip(
        frames_rgb=frames,
        frame_probs=probs,
        output_video_path=out_video,
        threshold=0.55,
        fps=2.0,
    )

    assert res_path.exists()
    assert res_path.stat().st_size > 0
