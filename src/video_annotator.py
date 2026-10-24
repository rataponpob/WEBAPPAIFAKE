from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import cv2
import imageio
import numpy as np


def draw_frame_overlay(
    frame_rgb: np.ndarray,
    frame_idx: int,
    total_frames: int,
    prob_fake: float,
    threshold: float = 0.55,
    fps: float = 30.0,
    is_sampled_frame: bool = False,
    borderline_band: Tuple[float, float] = (0.45, 0.65),
) -> np.ndarray:
    """วาดข้อมูลผลการวิเคราะห์ AI vs REAL ลงบนเฟรมวิดีโอ (RGB format)"""
    h, w = frame_rgb.shape[:2]
    annotated = frame_rgb.copy()

    is_fake = prob_fake >= threshold
    is_borderline = borderline_band[0] <= prob_fake <= borderline_band[1]

    # กำหนดชุดสี (RGB)
    if is_fake:
        status_color = (239, 68, 68)     # แดง (AI / FAKE)
        label_text = "AI / FAKE"
    else:
        status_color = (16, 185, 129)    # เขียว (REAL)
        label_text = "REAL VIDEO"

    # คำนวณสเกลตามขนาดของวิดีโอ
    scale = max(0.4, min(w, h) / 600.0)
    bar_height = int(max(48, 64 * scale))

    # 1. แถบ Overlay ด้านบนแบบกึ่งโปร่งใส (Alpha Blending)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (15, 23, 42), -1)  # Slate-900
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # 2. ป้าย Badge สถานะ (FAKE หรือ REAL)
    badge_w = int(140 * scale)
    badge_h = int(32 * scale)
    badge_x = int(12 * scale)
    badge_y = int((bar_height - badge_h) / 2)

    cv2.rectangle(
        annotated,
        (badge_x, badge_y),
        (badge_x + badge_w, badge_y + badge_h),
        status_color,
        -1,
    )
    cv2.putText(
        annotated,
        label_text,
        (badge_x + int(10 * scale), badge_y + int(22 * scale)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55 * scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # 3. ค่าความน่าจะเป็น prob(FAKE)
    prob_str = f"P(FAKE): {prob_fake * 100:.1f}%"
    prob_x = badge_x + badge_w + int(16 * scale)
    prob_y = badge_y + int(23 * scale)
    cv2.putText(
        annotated,
        prob_str,
        (prob_x, prob_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65 * scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 4. ข้อความ Borderline (ถ้ามี)
    if is_borderline:
        bl_text = "[Borderline]"
        cv2.putText(
            annotated,
            bl_text,
            (prob_x + int(165 * scale), prob_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50 * scale,
            (245, 158, 11),  # Amber
            1,
            cv2.LINE_AA,
        )

    # 5. ข้อมูลตำแหน่งเฟรมและเวลา (มุมขวาบน)
    current_sec = frame_idx / max(1.0, fps)
    mins = int(current_sec // 60)
    secs = current_sec % 60
    time_str = f"{mins:02d}:{secs:05.2f} (F#{frame_idx + 1}/{total_frames})"

    text_size = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, 1)[0]
    time_x = max(prob_x + int(250 * scale), w - text_size[0] - int(16 * scale))
    cv2.putText(
        annotated,
        time_str,
        (time_x, prob_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 * scale,
        (203, 213, 225),  # Slate-300
        1,
        cv2.LINE_AA,
    )

    # 6. แถบ Timeline ความคืบหน้าด้านล่างสุดของจอ (Bottom Progress Bar)
    progress_h = max(4, int(6 * scale))
    cv2.rectangle(
        annotated,
        (0, h - progress_h),
        (w, h),
        (30, 41, 59),
        -1,
    )
    progress_w = int(w * (frame_idx + 1) / max(1, total_frames))
    cv2.rectangle(
        annotated,
        (0, h - progress_h),
        (progress_w, h),
        status_color,
        -1,
    )

    return annotated


def create_annotated_video(
    input_video_path: Union[str, Path],
    output_video_path: Union[str, Path],
    frame_probs: List[float],
    threshold: float = 0.55,
    borderline_band: Tuple[float, float] = (0.45, 0.65),
    max_frames_to_process: int = 1800,  # จำกัด ~1 นาทีเพื่อความรวดเร็วในการเรนเดอร์
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Path:
    """สร้างไฟล์วิดีโอ MP4 ใหม่ที่มีการ Overlay ผลการตรวจจับ (FAKE / REAL)
    รายเฟรมตลอดทั้งคลิป พร้อมเข้ารหัสแบบ H.264 (YUV420p) เพื่อเล่นบนเว็บได้ทันที
    """
    in_path = str(input_video_path)
    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError(f"ไม่สามารถเปิดไฟล์วิดีโอได้: {in_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        if total_frames <= 0:
            total_frames = 1

        n_sampled = len(frame_probs)
        # คำนวณตำแหน่ง index ของเฟรมที่ถูกสุ่มตรวจตามสูตร Uniform Sampling
        sampled_idxs = [
            int(total_frames * f / (n_sampled + 1)) for f in range(1, n_sampled + 1)
        ]

        frames_to_write = min(total_frames, max_frames_to_process)
        writer = imageio.get_writer(
            str(out_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )

        for i in range(frames_to_write):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            # หาความน่าจะเป็นของเฟรมปัจจุบันโดยเทียบกับตำแหน่งเฟรมสุ่มที่ใกล้ที่สุด
            if n_sampled > 0:
                closest_idx = int(
                    np.argmin([abs(i - s_idx) for s_idx in sampled_idxs])
                )
                prob_fake = float(frame_probs[closest_idx])
            else:
                prob_fake = 0.0

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            annotated_frame = draw_frame_overlay(
                frame_rgb=frame_rgb,
                frame_idx=i,
                total_frames=total_frames,
                prob_fake=prob_fake,
                threshold=threshold,
                fps=fps,
                borderline_band=borderline_band,
            )

            writer.append_data(annotated_frame)

            if progress_callback and i % 15 == 0:
                progress_callback((i + 1) / frames_to_write)

        writer.close()
        if progress_callback:
            progress_callback(1.0)

    finally:
        cap.release()

    return out_path


def create_sampled_clip(
    frames_rgb: List[np.ndarray],
    frame_probs: List[float],
    output_video_path: Union[str, Path],
    threshold: float = 0.55,
    fps: float = 1.5,
    borderline_band: Tuple[float, float] = (0.45, 0.65),
) -> Path:
    """สร้างคลิปวิดีโอความเร็วต่ำ (1-2 fps) เฉพาะเฟรมที่ถูกสุ่มตรวจ
    เพื่อให้ผู้ใช้ตรวจดูรายละเอียดของแต่ละเฟรมได้อย่างชัดเจน
    """
    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(frames_rgb)
    if n_frames == 0:
        raise ValueError("ไม่มีเฟรมสำหรับสร้างวิดีโอ")

    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
    )

    for i, frame in enumerate(frames_rgb):
        prob = frame_probs[i] if i < len(frame_probs) else 0.0
        # ปรับขนาดให้เป็นเลขคู่ตามข้อกำหนดของ H.264 codec
        fh, fw = frame.shape[:2]
        if fw % 2 != 0 or fh % 2 != 0:
            frame = cv2.resize(frame, (fw - (fw % 2), fh - (fh % 2)))

        annotated = draw_frame_overlay(
            frame_rgb=frame,
            frame_idx=i,
            total_frames=n_frames,
            prob_fake=prob,
            threshold=threshold,
            fps=fps,
            is_sampled_frame=True,
            borderline_band=borderline_band,
        )
        writer.append_data(annotated)

    writer.close()
    return out_path
