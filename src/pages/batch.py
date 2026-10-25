import gc
import os
import tempfile
import time
from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import CONFIG
from src.frame_extractor import extract_frames
from src.model_loader import get_active_model_path, is_model_available, load_model
from src.predictor import predict_video


def guess_true_label(filename: str) -> str:
    """เดา label จริงจากชื่อไฟล์ (ถ้ามีคำว่า 'real' หรือ 'fake')"""
    name_lower = filename.lower()
    if "fake" in name_lower or "ai_" in name_lower or "sora" in name_lower:
        return "fake"
    if "real" in name_lower:
        return "real"
    return None


def render_batch_page():
    st.header("📦 ตรวจสอบหลายวิดีโอพร้อมกัน (Batch Video Analysis)")
    st.markdown(
        "อัปโหลดไฟล์วิดีโอหลายไฟล์พร้อมกันเพื่อวิเคราะห์แบบชุด "
        "ระบบจะประมวลผลทีละไฟล์ พร้อมสรุปสถิติและดาวน์โหลดผลลัพธ์เป็น CSV"
    )

    model_path = get_active_model_path()
    model_ready = is_model_available(model_path)

    if not model_ready:
        st.warning(
            f"⚠️ ยังไม่พบไฟล์โมเดลที่ `{model_path}` กรุณาวางไฟล์ `{model_path.name}` ก่อนเริ่มใช้งาน"
        )

    # แผงตั้งค่าการตรวจจับ (Inference Settings)
    with st.expander("⚙️ ตั้งค่าการตรวจจับ (Inference Settings)", expanded=False):
        c_set1, c_set2 = st.columns(2)
        with c_set1:
            cfg_n_frames = st.slider(
                "จำนวนเฟรมที่สุ่มตรวจ (N_FRAMES):",
                min_value=4,
                max_value=100,
                value=st.session_state.get("cfg_n_frames", CONFIG["N_FRAMES"]),
                step=2,
                help="จำนวนเฟรมต่อคลิปที่ตัดมาสุ่มตรวจ ยิ่งมากยิ่งเสถียรแต่ใช้เวลาเพิ่มขึ้น",
            )
            st.session_state["cfg_n_frames"] = cfg_n_frames
        with c_set2:
            cfg_thresh = st.slider(
                "เกณฑ์ตัดสิน FAKE (VOTE_THRESHOLD):",
                min_value=0.00,
                max_value=1.00,
                value=st.session_state.get("cfg_vote_threshold", CONFIG["VOTE_THRESHOLD"]),
                step=0.01,
                format="%.2f",
                help="ถ้า prob(FAKE) เฉลี่ย >= ค่านี้ ระบบจะตัดสินว่าเป็น FAKE",
            )
            st.session_state["cfg_vote_threshold"] = cfg_thresh

    uploaded_files = st.file_uploader(
        "เลือกหรือลากไฟล์วิดีโอหลายไฟล์มาวางที่นี่",
        type=[ext.replace(".", "") for ext in CONFIG["VIDEO_EXTS"]],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"📁 เลือกไฟล์ทั้งหมด {len(uploaded_files)} ไฟล์")

        n_frames = st.session_state.get("cfg_n_frames", CONFIG["N_FRAMES"])
        vote_thresh = st.session_state.get(
            "cfg_vote_threshold", CONFIG["VOTE_THRESHOLD"]
        )

        col_btn, _ = st.columns([2, 5])
        with col_btn:
            start_batch = st.button(
                "🚀 เริ่มวิเคราะห์ชุดวิดีโอ",
                type="primary",
                disabled=not model_ready,
                use_container_width=True,
            )

        if start_batch:
            model = load_model(model_path)
            rows = []
            progress_bar = st.progress(0, text="กำลังเตรียมเริ่มการวิเคราะห์...")
            status_box = st.empty()

            total_files = len(uploaded_files)

            for idx, file_obj in enumerate(uploaded_files):
                fname = file_obj.name
                true_lbl = guess_true_label(fname)
                percent = int((idx / total_files) * 100)
                progress_bar.progress(
                    percent, text=f"[{idx+1}/{total_files}] กำลังวิเคราะห์: {fname}"
                )

                temp_path = None
                t0 = time.time()
                try:
                    suffix = Path(fname).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file_obj.getvalue())
                        temp_path = tmp.name

                    frames = extract_frames(
                        temp_path, n_frames=n_frames, img_size=CONFIG["IMG_SIZE"]
                    )

                    if not frames:
                        rows.append(
                            {
                                "filename": fname,
                                "true_label": true_lbl,
                                "predicted_label": "ERROR (อ่านเฟรมไม่ได้)",
                                "mean_prob_fake": None,
                                "confidence": "N/A",
                                "n_frames_used": 0,
                                "elapsed_s": round(time.time() - t0, 2),
                            }
                        )
                    else:
                        res = predict_video(
                            model=model,
                            frames=frames,
                            threshold=vote_thresh,
                            class_index_fake=CONFIG["CLASS_INDEX_FAKE"],
                            borderline_band=CONFIG["BORDERLINE_BAND"],
                        )
                        rows.append(
                            {
                                "filename": fname,
                                "true_label": true_lbl,
                                "predicted_label": res.label.upper(),
                                "mean_prob_fake": round(res.mean_prob_fake, 4),
                                "confidence": res.confidence_level,
                                "n_frames_used": res.n_frames_used,
                                "elapsed_s": round(time.time() - t0, 2),
                            }
                        )

                        # บันทึกลง history
                        if "history" not in st.session_state:
                            st.session_state["history"] = []
                        st.session_state["history"].append(
                            {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "filename": fname,
                                "verdict": res.label.upper(),
                                "mean_prob_fake": res.mean_prob_fake,
                                "confidence": res.confidence_level,
                                "frames_used": res.n_frames_used,
                                "threshold": res.threshold,
                                "borderline": res.borderline,
                                "elapsed_s": round(time.time() - t0, 2),
                            }
                        )

                except Exception as ex:
                    rows.append(
                        {
                            "filename": fname,
                            "true_label": true_lbl,
                            "predicted_label": f"ERROR ({str(ex)[:30]})",
                            "mean_prob_fake": None,
                            "confidence": "N/A",
                            "n_frames_used": 0,
                            "elapsed_s": round(time.time() - t0, 2),
                        }
                    )
                finally:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass

            progress_bar.progress(100, text="✅ วิเคราะห์ชุดวิดีโอเสร็จสมบูรณ์!")
            gc.collect()

            df_results = pd.DataFrame(rows)
            st.session_state["latest_batch_df"] = df_results

        # แสดงตารางผลลัพธ์
        if "latest_batch_df" in st.session_state:
            df = st.session_state["latest_batch_df"]
            st.subheader("📊 ผลลัพธ์การวิเคราะห์แบบกลุ่ม (Batch Results)")

            # ตัวชี้วัดสรุป
            n_total = len(df)
            n_fake = sum(df["predicted_label"] == "FAKE")
            n_real = sum(df["predicted_label"] == "REAL")
            n_err = n_total - (n_fake + n_real)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ทั้งหมด (Total)", f"{n_total} ไฟล์")
            col2.metric("ทายว่า FAKE", f"{n_fake} ไฟล์", f"{(n_fake/n_total):.1%}")
            col3.metric("ทายว่า REAL", f"{n_real} ไฟล์", f"{(n_real/n_total):.1%}")
            col4.metric("ข้อผิดพลาด (Errors)", f"{n_err} ไฟล์")

            # ตาราง Dataframe
            st.dataframe(
                df.style.map(
                    lambda v: "background-color: #FEE2E2; font-weight: bold;"
                    if v == "FAKE"
                    else (
                        "background-color: #DCFCE7; font-weight: bold;"
                        if v == "REAL"
                        else ""
                    ),
                    subset=["predicted_label"],
                ),
                use_container_width=True,
            )

            # ประเมินผลหากมี true_label
            labeled = df[
                df["true_label"].notna()
                & df["predicted_label"].isin(["FAKE", "REAL"])
            ]
            if not labeled.empty:
                st.subheader("🎯 การประเมินผลเทียบกับ Ground Truth ในชื่อไฟล์")
                y_true = labeled["true_label"].str.upper()
                y_pred = labeled["predicted_label"]

                tp = sum((y_true == "FAKE") & (y_pred == "FAKE"))
                tn = sum((y_true == "REAL") & (y_pred == "REAL"))
                fp = sum((y_true == "REAL") & (y_pred == "FAKE"))
                fn = sum((y_true == "FAKE") & (y_pred == "REAL"))
                total_eval = tp + tn + fp + fn
                acc = (tp + tn) / total_eval if total_eval > 0 else 0

                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{acc:.1%}", f"{tp+tn}/{total_eval}")
                c2.metric(
                    "FAKE Recall",
                    f"{(tp / (tp + fn)):.1%}" if (tp + fn) > 0 else "N/A",
                )
                c3.metric(
                    "REAL Recall",
                    f"{(tn / (tn + fp)):.1%}" if (tn + fp) > 0 else "N/A",
                )

                st.markdown(
                    f"""
| | Predicted REAL | Predicted FAKE |
|---|---|---|
| **Actual REAL** | {tn} | {fp} |
| **Actual FAKE** | {fn} | {tp} |
                    """
                )

            # ปุ่ม Download CSV
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ ดาวน์โหลดผลลัพธ์เป็น CSV (predictions.csv)",
                data=csv_bytes,
                file_name=f"batch_predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
