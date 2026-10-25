import gc
import json
import os
import tempfile
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import CONFIG
from src.frame_extractor import extract_frames
from src.model_loader import (
    download_model_from_url,
    get_active_model_path,
    get_available_models,
    is_model_available,
    load_model,
    set_active_model_path,
)
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

    # ตรวจสอบและจัดการโมเดลในหน้านี้โดยตรง
    available_models = get_available_models()
    model_path = get_active_model_path()
    model_display_name = model_path.stem if model_path else "AI Model"
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
            if st.button("📥 โหลดโมเดล GitHub Release", type="primary", use_container_width=True, key="btn_batch_dl_model"):
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
            chosen = st.selectbox("สลับโมเดล:", options=m_names, index=cur_idx, key="batch_model_select")
            ch_path = next((m for m in available_models if m.name == chosen), model_path)
            if ch_path != model_path:
                set_active_model_path(ch_path)
                st.rerun()

    # แผงตั้งค่าการตรวจจับ (Inference Settings) แสดงให้เห็นชัดเจนทันที ไม่พับซ่อน
    st.markdown(
        "<div style='margin-top:4px;margin-bottom:2px;font-weight:600;font-size:0.95rem;color:#1e293b;'>"
        "⚙️ ตั้งค่าการตรวจจับ (Inference Settings)"
        "</div>",
        unsafe_allow_html=True,
    )
    c_set1, c_set2 = st.columns(2)
    with c_set1:
        cfg_n_frames = st.slider(
            "🎞️ จำนวนเฟรมที่สุ่มตรวจ (N_FRAMES):",
            min_value=4,
            max_value=100,
            value=st.session_state.get("cfg_n_frames", CONFIG["N_FRAMES"]),
            step=2,
            help="จำนวนเฟรมต่อคลิปที่ตัดมาสุ่มตรวจ ยิ่งมากยิ่งเสถียรแต่ใช้เวลาเพิ่มขึ้น",
        )
        st.session_state["cfg_n_frames"] = cfg_n_frames
    with c_set2:
        cfg_thresh = st.slider(
            "🎯 เกณฑ์ตัดสิน FAKE (VOTE_THRESHOLD):",
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
                    percent, text=f"[{idx+1}/{total_files}] ⚡ กำลังวิเคราะห์ด้วย {model_display_name}: {fname}"
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
                                "model_name": model_display_name,
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
                                "model_name": model_display_name,
                                "true_label": true_lbl,
                                "predicted_label": res.label.upper(),
                                "mean_prob_fake": round(res.mean_prob_fake, 4),
                                "confidence": res.confidence_level,
                                "borderline": res.borderline,
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
                                "model_name": model_display_name,
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
                            "model_name": model_display_name,
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

            progress_bar.progress(100, text=f"✅ วิเคราะห์ชุดวิดีโอด้วย {model_display_name} เสร็จสมบูรณ์!")
            gc.collect()

            df_results = pd.DataFrame(rows)
            st.session_state["latest_batch_df"] = df_results
            st.session_state["latest_batch_model"] = model_display_name

        # แสดงผลลัพธ์การวิเคราะห์แบบกลุ่ม (Rich & Clean Batch Results)
        if "latest_batch_df" in st.session_state:
            df = st.session_state["latest_batch_df"]
            used_m = st.session_state.get("latest_batch_model", model_display_name)

            st.divider()
            st.subheader("📊 ผลลัพธ์การวิเคราะห์แบบกลุ่ม (Batch Results)")
            st.caption(
                f"🤖 โมเดลที่ใช้: **{used_m}** • สุ่มตรวจ: **{n_frames}** เฟรม/คลิป • เกณฑ์ตัดสิน FAKE: **{vote_thresh:.2f}**"
            )

            # 1. กล่องสรุปสถิติภาพรวม (KPI Cards)
            n_total = len(df)
            n_fake = sum(df["predicted_label"] == "FAKE")
            n_real = sum(df["predicted_label"] == "REAL")
            n_err = sum(df["predicted_label"].str.startswith("ERROR", na=False))
            n_border = sum(df.get("borderline", [False] * n_total))
            total_time = sum(df.get("elapsed_s", [0] * n_total))

            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("📁 ทั้งหมด", f"{n_total} ไฟล์")
            kpi2.metric(
                "🔴 ตรวจพบ FAKE (AI)",
                f"{n_fake} ไฟล์",
                f"{(n_fake/n_total*100):.1f}%" if n_total > 0 else "0%",
            )
            kpi3.metric(
                "🟢 ตรวจพบ REAL (จริง)",
                f"{n_real} ไฟล์",
                f"{(n_real/n_total*100):.1f}%" if n_total > 0 else "0%",
            )
            kpi4.metric("⚠️ ก้ำกึ่ง (Borderline)", f"{n_border} ไฟล์")
            kpi5.metric("⏱️ เวลารวม", f"{total_time:.1f} วินาที")

            # 2. กราฟภาพรวม (Visual Insights)
            c_chart1, c_chart2 = st.columns([1, 1], gap="medium")
            with c_chart1:
                labels = ["FAKE (AI)", "REAL (จริง)"]
                values = [n_fake, n_real]
                colors = ["#EF4444", "#10B981"]
                if n_err > 0:
                    labels.append("Error")
                    values.append(n_err)
                    colors.append("#94A3B8")

                fig_donut = go.Figure(
                    data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.55,
                            marker_colors=colors,
                            textinfo="label+percent",
                            hoverinfo="label+value+percent",
                        )
                    ]
                )
                fig_donut.update_layout(
                    title="สัดส่วนผลวิเคราะห์ (FAKE vs REAL)",
                    margin=dict(l=10, r=10, t=35, b=10),
                    height=270,
                    showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with c_chart2:
                valid_df = df[df["mean_prob_fake"].notna()].sort_values("mean_prob_fake", ascending=True)
                if not valid_df.empty:
                    bar_colors = [
                        "#EF4444" if p >= vote_thresh else "#10B981"
                        for p in valid_df["mean_prob_fake"]
                    ]
                    short_names = [
                        name[:22] + "..." if len(name) > 25 else name
                        for name in valid_df["filename"]
                    ]
                    fig_bar = go.Figure()
                    fig_bar.add_trace(
                        go.Bar(
                            y=short_names,
                            x=[p * 100 for p in valid_df["mean_prob_fake"]],
                            orientation="h",
                            marker_color=bar_colors,
                            text=[f"{p*100:.1f}%" for p in valid_df["mean_prob_fake"]],
                            textposition="auto",
                            hovertemplate="<b>%{y}</b><br>AI Prob: %{x:.1f}%<extra></extra>",
                        )
                    )
                    fig_bar.add_vline(
                        x=vote_thresh * 100,
                        line_dash="dash",
                        line_color="#F59E0B",
                        annotation_text=f"เกณฑ์ ({vote_thresh*100:.0f}%)",
                        annotation_position="top right",
                    )
                    fig_bar.update_layout(
                        title="อันดับคะแนน AI Probability (% ยิ่งสูงยิ่งเป็น AI)",
                        xaxis_title="AI Probability (%)",
                        xaxis=dict(range=[0, 105]),
                        margin=dict(l=10, r=10, t=35, b=10),
                        height=max(270, len(valid_df) * 36),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # 3. ตารางแสดงผลลัพธ์แบบจัดรูปแบบอ่านง่าย (Formatted Table)
            st.markdown("##### 📋 ตารางสรุปรายละเอียดรายไฟล์")

            col_filter, col_search = st.columns([2, 3])
            with col_filter:
                filter_choice = st.selectbox(
                    "กรองแสดงผลลัพธ์:",
                    options=[
                        "ทั้งหมด (All)",
                        "🔴 เฉพาะ FAKE (AI)",
                        "🟢 เฉพาะ REAL (จริง)",
                        "⚠️ เฉพาะ ก้ำกึ่ง (Borderline)",
                    ],
                    key="batch_filter_choice",
                )
            with col_search:
                search_query = st.text_input(
                    "🔍 ค้นหาชื่อไฟล์:",
                    placeholder="พิมพ์ชื่อไฟล์ที่ต้องการค้นหา...",
                    key="batch_search_query",
                )

            display_df = df.copy()
            if filter_choice == "🔴 เฉพาะ FAKE (AI)":
                display_df = display_df[display_df["predicted_label"] == "FAKE"]
            elif filter_choice == "🟢 เฉพาะ REAL (จริง)":
                display_df = display_df[display_df["predicted_label"] == "REAL"]
            elif filter_choice == "⚠️ เฉพาะ ก้ำกึ่ง (Borderline)":
                display_df = display_df[display_df.get("borderline", False) == True]

            if search_query.strip():
                display_df = display_df[
                    display_df["filename"].str.contains(search_query.strip(), case=False, na=False)
                ]

            view_table = pd.DataFrame()
            view_table["📁 ไฟล์วิดีโอ"] = display_df["filename"]
            view_table["🎯 ผลการตรวจ"] = display_df["predicted_label"].apply(
                lambda x: "🔴 FAKE (AI)" if x == "FAKE" else ("🟢 REAL (จริง)" if x == "REAL" else f"⚠️ {x}")
            )
            view_table["📊 ความน่าจะเป็น AI"] = display_df["mean_prob_fake"].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"
            )
            view_table["🛡️ ความมั่นใจ"] = display_df["confidence"]
            view_table["🎞️ เฟรม"] = display_df["n_frames_used"].apply(lambda x: f"{x} เฟรม")
            view_table["⏱️ เวลา"] = display_df["elapsed_s"].apply(lambda x: f"{x:.2f} วินาที")

            if "true_label" in display_df.columns and display_df["true_label"].notna().any():
                view_table["🏷️ ป้ายกำกับจริง"] = display_df["true_label"].apply(
                    lambda x: str(x).upper() if pd.notna(x) else "-"
                )

            st.dataframe(
                view_table,
                use_container_width=True,
                hide_index=True,
            )

            # 4. ประเมินผล Ground Truth & Confusion Matrix (ถ้ามีข้อมูลในชื่อไฟล์)
            labeled = df[
                df["true_label"].notna()
                & df["predicted_label"].isin(["FAKE", "REAL"])
            ]
            if not labeled.empty:
                st.markdown("##### 🎯 การประเมินผลเทียบกับ Ground Truth ในชื่อไฟล์")
                y_true = labeled["true_label"].str.upper()
                y_pred = labeled["predicted_label"]

                tp = sum((y_true == "FAKE") & (y_pred == "FAKE"))
                tn = sum((y_true == "REAL") & (y_pred == "REAL"))
                fp = sum((y_true == "REAL") & (y_pred == "FAKE"))
                fn = sum((y_true == "FAKE") & (y_pred == "REAL"))
                total_eval = tp + tn + fp + fn
                acc = (tp + tn) / total_eval if total_eval > 0 else 0
                prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0
                recall_fake = (tp / (tp + fn)) if (tp + fn) > 0 else 0
                recall_real = (tn / (tn + fp)) if (tn + fp) > 0 else 0

                c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                c_m1.metric("🎯 ความแม่นยำ (Accuracy)", f"{acc:.1%}", f"{tp+tn}/{total_eval} ไฟล์")
                c_m2.metric("🎯 FAKE Precision", f"{prec:.1%}")
                c_m3.metric("🔴 FAKE Recall (ตรวจจับ AI)", f"{recall_fake:.1%}")
                c_m4.metric("🟢 REAL Recall (ตรวจจับของจริง)", f"{recall_real:.1%}")

                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px; margin-top:8px; margin-bottom:12px;">
                        <div style="font-weight:600; margin-bottom:8px; font-size:0.92rem; color:#334155;">ตาราง Confusion Matrix:</div>
                        <div style="display:grid; grid-template-columns: 140px 1fr 1fr; gap:8px; text-align:center; font-size:0.88rem;">
                            <div style="font-weight:bold; background:#E2E8F0; padding:8px; border-radius:6px;">จริง \\ ทาย</div>
                            <div style="font-weight:bold; background:#E0F2FE; color:#0369A1; padding:8px; border-radius:6px;">ทายว่า REAL (จริง)</div>
                            <div style="font-weight:bold; background:#FEE2E2; color:#B91C1C; padding:8px; border-radius:6px;">ทายว่า FAKE (AI)</div>
                            <div style="font-weight:bold; background:#F1F5F9; padding:10px; border-radius:6px; display:flex; align-items:center; justify-content:center;">ของจริง (REAL)</div>
                            <div style="background:#DCFCE7; color:#15803D; font-weight:bold; font-size:1.1rem; padding:10px; border-radius:6px; border:1px solid #86EFAC;">True Real (TN): {tn}</div>
                            <div style="background:#FEE2E2; color:#B91C1C; font-weight:bold; font-size:1.1rem; padding:10px; border-radius:6px; border:1px solid #FCA5A5;">False Fake (FP): {fp}</div>
                            <div style="font-weight:bold; background:#F1F5F9; padding:10px; border-radius:6px; display:flex; align-items:center; justify-content:center;">คลิป AI (FAKE)</div>
                            <div style="background:#FEE2E2; color:#B91C1C; font-weight:bold; font-size:1.1rem; padding:10px; border-radius:6px; border:1px solid #FCA5A5;">False Real (FN): {fn}</div>
                            <div style="background:#DCFCE7; color:#15803D; font-weight:bold; font-size:1.1rem; padding:10px; border-radius:6px; border:1px solid #86EFAC;">True Fake (TP): {tp}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 5. ปุ่มดาวน์โหลดผลลัพธ์ (CSV & JSON)
            st.markdown("##### 📥 ส่งออกข้อมูลผลวิเคราะห์")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ ดาวน์โหลดรายงานแบบ CSV (predictions.csv)",
                    data=csv_bytes,
                    file_name=f"batch_predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_dl2:
                json_bytes = json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button(
                    label="⬇️ ดาวน์โหลดรายงานแบบ JSON (predictions.json)",
                    data=json_bytes,
                    file_name=f"batch_predictions_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
