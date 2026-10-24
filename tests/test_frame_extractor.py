import numpy as np
from src.frame_extractor import extract_frames


def test_sampling_indices_formula():
    """ทดสอบสูตรคำนวณ Uniform sampling index ว่าตรงกับสมการในโน้ตบุ๊ก"""
    total = 100
    n_frames = 16

    idxs = [int(total * f / (n_frames + 1)) for f in range(1, n_frames + 1)]

    # ต้องได้ 16 ค่า เรียงลำดับจากน้อยไปมาก
    assert len(idxs) == 16
    assert idxs == sorted(idxs)
    # ต้องไม่แตะ index แรกสุด (0) หรือ index เกิน total
    assert idxs[0] > 0
    assert idxs[-1] < total
    # ตรวจสอบค่าตัวอย่าง
    assert idxs[0] == int(100 * 1 / 17)
    assert idxs[-1] == int(100 * 16 / 17)


def test_extract_frames_nonexistent_file():
    """ทดสอบกรณีส่งไฟล์ที่ไม่มีอยู่จริง ต้องคืน list ว่างและไม่ crash"""
    res = extract_frames("non_existent_video_path.mp4", n_frames=16)
    assert res == []
