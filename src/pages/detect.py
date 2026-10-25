import json
import os
import tempfile
import time
from pathlib import Path
import streamlit as st

from src.config import CONFIG
from src.frame_extractor import extract_frames, extract_frames_dual
from src.model_loader import (
    download_model_from_url,
    get_active_model_path,
    get_available_models,
    is_model_available,
    load_model,
    set_active_model_path,
)
from src.predictor import PredictionResult, predict_video
from src.video_annotator import create_annotated_video, create_sampled_clip
from src.visuals import (
    plot_frame_probs,
    plot_gauge,
    render_frame_gallery,
    render_verdict_card,
)


def render_detect_page():
    st.header("🎯 ตรวจสอบวิดีโอเดี่ยว (Single Video Detection)")
    st.caption(
        "อัปโหลดไฟล์วิดีโอเพื่อวิเคราะห์ด้วยโมเดล ResNet50 (Transfer Learning) "
        "สุ่มตรวจเฟรมแบบกระจายสม่ำเสมอตลอดคลิป"
    )

    # ตรวจสอบและจัดการโมเดลในหน้านี้โดยตรง
    available_models = get_available_models()
    model_path = get_active_model_path()
    model_ready = is_model_available(model_path)

    col_model_status, col_model_ctrl = st.columns([3, 1])
    with col_model_status:
        if model_ready:
            file_mb = model_path.stat().st_size / (1024 * 1024)
            st.markdown(
                f"<div style='background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);border-radius:8px;padding:6px 12px;font-size:0.86rem;margin-bottom:8px;'>"
                f"🤖 โมเดลที่ใช้งาน: <b style='color:#059669;'>{model_path.name}</b> "
                f"<span style='color:#6b7280;'>({file_mb:.1f} MB)</span> "
                f"<span style='color:#10b981;font-weight:600;margin-left:8px;'>● พร้อมทำงาน</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);border-radius:8px;padding:6px 12px;font-size:0.86rem;color:#dc2626;margin-bottom:8px;'>"
                f"⚠️ ยังไม่พบไฟล์โมเดลในระบบ กรุณากดดาวน์โหลดโมเดลจาก GitHub Release</div>",
                unsafe_allow_html=True,
            )

    with col_model_ctrl:
        if not model_ready:
            default_url = CONFIG.get("DEFAULT_MODEL_URL", "")
            if st.button("📥 โหลดโมเดล GitHub Release", type="primary", use_container_width=True, key="btn_detect_dl_model"):
                pbar = st.progress(0, text="กำลังเตรียมดาวน์โหลด...")
                s_txt = st.empty()
                with st.spinner("กำลังดาวน์โหลดโมเดล..."):
                    ok, msg = download_model_from_url(default_url, progress_bar=pbar, status_text=s_txt)
                    if ok:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
        elif len(available_models) > 1:
            m_names = [m.name for m in available_models]
            cur_idx = m_names.index(model_path.name) if model_path.name in m_names else 0
            chosen = st.selectbox("สลับโมเดล:", options=m_names, index=cur_idx, key="detect_model_select")
            ch_path = next((m for m in available_models if m.name == chosen), model_path)
            if ch_path != model_path:
                set_active_model_path(ch_path)
                st.rerun()

    # ตัวอัปโหลดไฟล์
    uploaded_file = st.file_uploader(
        "เลือกหรือลากไฟล์วิดีโอมาวางที่นี่ (รองรับ mp4, mov, avi, webm, mkv)",
        type=[ext.replace(".", "") for ext in CONFIG["VIDEO_EXTS"]],
        help=f"ขนาดไฟล์สูงสุด {CONFIG['MAX_UPLOAD_MB']} MB",
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > CONFIG["MAX_UPLOAD_MB"]:
            st.error(
                f"❌ ไฟล์มีขนาด {file_size_mb:.1f} MB เกินขีดจำกัดที่กำหนดไว้ "
                f"({CONFIG['MAX_UPLOAD_MB']} MB) กรุณาลดขนาดหรือตัดคลิปให้สั้นลง"
            )
            return

        # ค่า Config จาก Sidebar (หรือ session state)
        n_frames = st.session_state.get("cfg_n_frames", CONFIG["N_FRAMES"])
        vote_thresh = st.session_state.get(
            "cfg_vote_threshold", CONFIG["VOTE_THRESHOLD"]
        )

        # จัดส่วนพรีวิวและปุ่มเริ่มวิเคราะห์เป็น 2 คอลัมน์กะทัดรัด
        c_prev, c_info = st.columns([1, 1], gap="medium")
        with c_prev:
            st.video(uploaded_file)
        with c_info:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                    <p style="margin: 0 0 4px 0;"><strong>📁 ไฟล์:</strong> <code>{uploaded_file.name}</code></p>
                    <p style="margin: 0 0 4px 0;"><strong>📦 ขนาด:</strong> {file_size_mb:.2f} MB</p>
                    <p style="margin: 0 0 4px 0;"><strong>🎞️ สุ่มตรวจ:</strong> {n_frames} เฟรม</p>
                    <p style="margin: 0;"><strong>🎯 เกณฑ์ตัดสิน:</strong> {vote_thresh:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            start_btn = st.button(
                "🚀 เริ่มวิเคราะห์วิดีโอ",
                type="primary",
                disabled=not model_ready,
                use_container_width=True,
            )

        if start_btn:
            # Inference Pipeline พร้อม Temporary File
            temp_path = None
            try:
                with st.status("🔍 กำลังประมวลผลวิดีโอ...", expanded=True) as status:
                    t_start = time.time()

                    # ขั้นตอนที่ 1: บันทึกลง temporary file
                    status.update(label="1/4 💾 กำลังบันทึกไฟล์ชั่วคราว...")
                    suffix = Path(uploaded_file.name).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_path = tmp.name

                    # ขั้นตอนที่ 2: โหลดโมเดล
                    status.update(label="2/4 🧠 กำลังโหลดโมเดล ResNet50...")
                    model = load_model(model_path)

                    # ขั้นตอนที่ 3: สุ่มตัดเฟรม uniform sampling
                    status.update(
                        label=f"3/4 🎞️ กำลังสุ่มตัด {n_frames} เฟรมแบบ Uniform..."
                    )
                    model_frames, display_frames = extract_frames_dual(
                        temp_path, n_frames=n_frames, img_size=CONFIG["IMG_SIZE"]
                    )

                    if not model_frames:
                        status.update(label="❌ เกิดข้อผิดพลาด", state="error")
                        st.error(
                            "❌ อ่านวิดีโอไม่สำเร็จ (ไฟล์อาจเสียหายหรือ codec ไม่รองรับ) "
                            "ลองแปลงเป็น .mp4 (H.264) แล้วอัปโหลดใหม่อีกครั้ง"
                        )
                        return

                    if len(model_frames) < min(8, n_frames):
                        st.warning(
                            f"⚠️ อ่านได้เพียง {len(model_frames)}/{n_frames} เฟรม "
                            "(คลิปสั้นมากหรือเฟรมเสียหายบางส่วน) ผลวิเคราะห์อาจมีความคลาดเคลื่อน"
                        )

                    # ขั้นตอนที่ 4: Predict (batch เดียว)
                    status.update(
                        label=f"4/4 ⚡ กำลังทำนายด้วย ResNet50 ({len(model_frames)} เฟรม)..."
                    )
                    result = predict_video(
                        model=model,
                        frames=model_frames,
                        threshold=vote_thresh,
                        class_index_fake=CONFIG["CLASS_INDEX_FAKE"],
                        borderline_band=CONFIG["BORDERLINE_BAND"],
                    )

                    # สร้างคลิปวิดีโอระบุเฟรมสุ่มพร้อม Overlay (รักษา Aspect Ratio เดิม ไม่บีบภาพแบน)
                    sampled_clip_bytes = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_sc:
                            tmp_sc_path = tmp_sc.name
                        create_sampled_clip(
                            frames_rgb=display_frames,
                            frame_probs=result.frame_probs,
                            output_video_path=tmp_sc_path,
                            threshold=vote_thresh,
                            fps=1.5,
                            borderline_band=CONFIG["BORDERLINE_BAND"],
                        )
                        with open(tmp_sc_path, "rb") as scf:
                            sampled_clip_bytes = scf.read()
                        os.remove(tmp_sc_path)
                    except Exception as clip_err:
                        print(f"Error creating sampled clip: {clip_err}")

                    elapsed = time.time() - t_start
                    status.update(
                        label=f"✅ วิเคราะห์เสร็จสิ้นใน {elapsed:.2f} วินาที",
                        state="complete",
                    )

                st.divider()

                # บันทึกลง Session State เพื่อคงผลลัพธ์ไว้เมื่อ rerun
                st.session_state["latest_result"] = {
                    "filename": uploaded_file.name,
                    "result": result,
                    "frames": display_frames,
                    "elapsed": elapsed,
                    "sampled_clip_bytes": sampled_clip_bytes,
                    "raw_video_bytes": uploaded_file.getvalue(),
                }

                # เพิ่มลง History
                if "history" not in st.session_state:
                    st.session_state["history"] = []

                st.session_state["history"].append(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded_file.name,
                        "verdict": result.label.upper(),
                        "mean_prob_fake": result.mean_prob_fake,
                        "confidence": result.confidence_level,
                        "frames_used": result.n_frames_used,
                        "threshold": result.threshold,
                        "borderline": result.borderline,
                        "elapsed_s": round(elapsed, 2),
                    }
                )

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดขณะประมวลผล: {e}")
            finally:
                # ปฏิบัติตามหลัก Stateless: ลบ Temp File เสมอ
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        # แสดงผลลัพธ์ล่าสุดหากมีอยู่ใน session
        if "latest_result" in st.session_state and (
            st.session_state["latest_result"]["filename"] == uploaded_file.name
        ):
            saved = st.session_state["latest_result"]
            res: PredictionResult = saved["result"]
            frames = saved["frames"]

            # 1. แสดงการ์ดผลลัพธ์หลัก
            render_verdict_card(res)

            # 2. คอลัมน์คู่: ซ้ายเป็นวิดีโอผลวิเคราะห์ ขวาเป็น Gauge & Bar Chart
            st.divider()
            c_vid, c_stat = st.columns([1, 1], gap="medium")

            with c_vid:
                st.markdown("##### 🎬 วิดีโอระบุผลวิเคราะห์ (Annotated Video)")
                tab_vid_sampled, tab_vid_full = st.tabs(
                    [
                        "🎥 เฟรมสุ่ม (1.5 fps)",
                        "🎞️ วิดีโอตัวเต็ม",
                    ]
                )

                with tab_vid_sampled:
                    if saved.get("sampled_clip_bytes"):
                        st.video(saved["sampled_clip_bytes"])
                        st.caption("ℹ️ แสดง 16 เฟรมที่ผ่าน ResNet50 พร้อมป้ายระบุ FAKE/REAL รายเฟรม")
                        st.download_button(
                            label="⬇️ ดาวน์โหลดวิดีโอเฟรมสุ่ม (MP4)",
                            data=saved["sampled_clip_bytes"],
                            file_name=f"annotated_sampled_{Path(saved['filename']).stem}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )
                    else:
                        st.info("ไม่มีข้อมูลวิดีโอตรวจเฟรมสุ่ม")

                with tab_vid_full:
                    if saved.get("full_annotated_bytes"):
                        st.video(saved["full_annotated_bytes"])
                        st.caption("ℹ️ วิดีโอต้นฉบับความเร็วปกติ พร้อม Overlay ตลอดทั้งคลิป")
                        st.download_button(
                            label="⬇️ ดาวน์โหลดวิดีโอตัวเต็ม (MP4)",
                            data=saved["full_annotated_bytes"],
                            file_name=f"annotated_full_{Path(saved['filename']).stem}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )
                    else:
                        st.caption("💡 เรนเดอร์วิดีโอต้นฉบับความเร็วปกติพร้อมแถบระบุเฟรมตลอดคลิป")
                        if st.button("🚀 เรนเดอร์วิดีโอตัวเต็มพร้อม Overlay", type="secondary", use_container_width=True):
                            with st.status("🎬 กำลังเรนเดอร์...", expanded=False) as v_status:
                                temp_in = None
                                temp_out = None
                                try:
                                    suffix = Path(saved["filename"]).suffix or ".mp4"
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as ti:
                                        ti.write(saved["raw_video_bytes"])
                                        temp_in = ti.name
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as to:
                                        temp_out = to.name

                                    create_annotated_video(
                                        input_video_path=temp_in,
                                        output_video_path=temp_out,
                                        frame_probs=res.frame_probs,
                                        threshold=res.threshold,
                                        borderline_band=CONFIG["BORDERLINE_BAND"],
                                    )

                                    with open(temp_out, "rb") as of:
                                        saved["full_annotated_bytes"] = of.read()

                                    v_status.update(label="✅ สำเร็จ!", state="complete")
                                    st.rerun()
                                except Exception as re_err:
                                    v_status.update(label=f"❌ เกิดข้อผิดพลาด: {re_err}", state="error")
                                    st.error(f"ไม่สามารถเรนเดอร์: {re_err}")
                                finally:
                                    for tp in (temp_in, temp_out):
                                        if tp and os.path.exists(tp):
                                            try:
                                                os.remove(tp)
                                            except Exception:
                                                pass

            with c_stat:
                st.markdown("##### 📊 ผลการประเมินความน่าจะเป็น")
                st.plotly_chart(
                    plot_gauge(res.mean_prob_fake, res.threshold),
                    use_container_width=True,
                )
                st.plotly_chart(
                    plot_frame_probs(res.frame_probs, res.threshold),
                    use_container_width=True,
                )

            # 3. Frame Gallery
            st.divider()
            render_frame_gallery(frames, res.frame_probs, threshold=res.threshold)

            # 4. ปุ่มดาวน์โหลดรายงาน JSON
            report_data = {
                "filename": saved["filename"],
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": saved["elapsed"],
                "prediction": res.to_dict(),
            }
            st.download_button(
                label="⬇️ ดาวน์โหลดรายงานผลลัพธ์ (JSON)",
                data=json.dumps(report_data, ensure_ascii=False, indent=2),
                file_name=f"ai_detector_{Path(saved['filename']).stem}.json",
                mime="application/json",
            )
