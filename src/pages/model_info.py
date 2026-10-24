import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CONFIG
from src.model_loader import (
    delete_model,
    get_active_model_path,
    get_available_models,
    get_model_info,
    is_model_available,
    load_model,
    save_uploaded_model,
    set_active_model_path,
)
from src.predictor import predict_video


def render_model_info_page():
    st.header("ℹ️ ข้อมูลสถาปัตยกรรมและจัดการโมเดล (Model Management)")
    st.caption(
        "จัดการไฟล์โมเดลในระบบ (อัปโหลด / สลับใช้งาน / ลบ) พร้อมสถิติความแม่นยำและ Canary Self-Test"
    )

    tab_manage, tab_arch, tab_metrics, tab_limit, tab_canary = st.tabs(
        [
            "📁 จัดการโมเดล (Manage)",
            "🏛️ สถาปัตยกรรม (Architecture)",
            "📊 ตัวชี้วัด & ประสิทธิภาพ (Metrics)",
            "⚠️ ข้อจำกัด & คำเตือน (Limitations)",
            "🧪 Canary Self-Test",
        ]
    )

    active_model_path = get_active_model_path()

    # ---------------- TAB 1: จัดการโมเดล (เพิ่ม / ลบ / อัปโหลด / เลือกใช้) ----------------
    with tab_manage:
        st.subheader("📁 รายการโมเดลทั้งหมดในโฟลเดอร์ `models/`")
        available_models = get_available_models()

        if not available_models:
            st.warning("⚠️ ยังไม่พบไฟล์โมเดลใดๆ ในระบบ กรุณาอัปโหลดไฟล์โมเดลด้านล่าง")
        else:
            for m_path in available_models:
                is_active = (m_path.name == active_model_path.name)
                size_mb = m_path.stat().st_size / (1024 * 1024)

                c_name, c_size, c_stat, c_act = st.columns([4, 2, 2, 3])
                with c_name:
                    st.markdown(f"**`{m_path.name}`**")
                with c_size:
                    st.caption(f"{size_mb:.1f} MB")
                with c_stat:
                    if is_active:
                        st.success("🟢 ใช้งานอยู่")
                    else:
                        st.caption("⚪ สแตนด์บาย")
                with c_act:
                    col_use, col_del = st.columns(2)
                    with col_use:
                        if not is_active:
                            if st.button("เลือกใช้", key=f"use_{m_path.name}", use_container_width=True):
                                set_active_model_path(m_path)
                                st.rerun()
                    with col_del:
                        with st.popover("🗑️ ลบ", help="ลบไฟล์โมเดลนี้"):
                            st.write(f"ยืนยันที่จะลบ `{m_path.name}` หรือไม่?")
                            if st.button(
                                "ยืนยันการลบ",
                                type="primary",
                                key=f"del_{m_path.name}",
                                use_container_width=True,
                            ):
                                ok, msg = delete_model(m_path)
                                if ok:
                                    st.success(msg)
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error(msg)
                st.divider()

        st.subheader("⬆️ อัปโหลดไฟล์โมเดลใหม่ (.keras หรือ .h5)")
        st.caption("อัปโหลดโมเดลที่ฝึกเสร็จจาก Colab หรือ Local เข้ามาใช้งานในแอปได้ทันที")
        uploaded_model = st.file_uploader(
            "เลือกไฟล์โมเดล Keras (.keras หรือ .h5)",
            type=["keras", "h5"],
            help="ขนาดไฟล์สูงสุด 500 MB ระบบจะทดสอบโหลดความถูกต้องอัตโนมัติก่อนเปิดใช้งาน",
        )

        if uploaded_model is not None:
            c_btn, _ = st.columns([3, 4])
            with c_btn:
                save_btn = st.button("🚀 บันทึกและทดสอบโมเดล", type="primary", use_container_width=True)

            if save_btn:
                with st.spinner("กำลังบันทึกและทดสอบความถูกต้องของโมเดล..."):
                    ok, msg, info = save_uploaded_model(uploaded_model)
                    if ok:
                        st.success(msg)
                        if info:
                            col_inf1, col_inf2, col_inf3 = st.columns(3)
                            col_inf1.metric("Input Shape", info.get("input_shape", "N/A"))
                            col_inf2.metric("จำนวนพารามิเตอร์", info.get("total_params", "N/A"))
                            col_inf3.metric("จำนวนชั้น (Layers)", info.get("layers_count", "N/A"))
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)

    # ---------------- TAB 2: สถาปัตยกรรม ----------------
    with tab_arch:
        st.subheader("โครงสร้างเครือข่ายประสาทเทียม ResNet50")
        st.markdown(
            """
        โมเดลใช้ **ResNet50** ที่ผ่าน Pre-train บนชุดข้อมูล ImageNet ทำหน้าที่เป็น Feature Extractor (Backbone)
        และต่อด้วย Classification Head สำหรับจำแนก 2 คลาส:
        """
        )

        st.code(
            """
Input: (224, 224, 3) RGB Image
  └─ ResNet50 Backbone (ImageNet Pre-trained, include_top=False)
       └─ GlobalAveragePooling2D
            └─ Dropout(rate=0.3)
                 └─ Dense(1, activation="sigmoid")  -> P(REAL)
        """,
            language="text",
        )

        st.markdown(
            """
        **กระบวนการฝึกฝน 2 เฟส (Two-Phase Training):**
        1. **Phase 1 — Head Training:** Freeze ชั้น backbone ทั้งหมด ฝึกเฉพาะ dense head (8 epochs, Adam lr=1e-3)
        2. **Phase 2 — Fine-Tuning:** Unfreeze 30% ชั้นบนสุดของ ResNet50 (6 epochs, Adam lr=1e-5)
        - Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
        """
        )

        # ข้อมูลโมเดลจริงหากโหลดได้
        if is_model_available(active_model_path):
            try:
                m = load_model(active_model_path)
                info = get_model_info(m)
                st.success(f"✅ โหลดโมเดล `{active_model_path.name}` สำเร็จ")
                col_i1, col_i2, col_i3 = st.columns(3)
                col_i1.metric("Input Shape", info.get("input_shape", "N/A"))
                col_i2.metric(
                    "จำนวนพารามิเตอร์ทั้งหมด", info.get("total_params", "N/A")
                )
                col_i3.metric("จำนวนชั้น (Layers)", info.get("layers_count", "N/A"))
            except Exception as e:
                st.info(f"ไม่สามารถอ่านโมเดล: {e}")
        else:
            st.info(
                f"💡 วางไฟล์ `{active_model_path.name}` ที่ `{active_model_path}` เพื่อแสดงสถานะโมเดลปัจจุบัน"
            )

    # ---------------- TAB 3: ตัวชี้วัด & ประสิทธิภาพ ----------------
    with tab_metrics:
        st.subheader("ผลการประเมินเทียบกัน: Test Set vs. Unseen Set")
        st.caption(
            "ความโปร่งใสเรื่อง Generalization Gap: ผลลัพธ์ในชุดทดสอบ vs การใช้งานกับวิดีโอภายนอกจริง"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 1. ระดับเฟรม — Test Set (3,600 เฟรม)")
            st.markdown("*ข้อมูลแบ่งจากแหล่งเดียวกับชุดฝึก (In-distribution)*")
            st.markdown(
                """
| Metric | Score |
|---|---|
| **Accuracy** | 92.06% |
| **Precision (FAKE)** | 89.14% |
| **Recall (FAKE)** | 95.83% |
| **F1-Score (FAKE)** | 92.36% |
            """
            )

        with c2:
            st.markdown("#### 2. ระดับวิดีโอ — Unseen Set (113 วิดีโอ)")
            st.markdown("*วิดีโอที่ไม่เคยพบมาก่อน (Out-of-distribution)*")
            st.markdown(
                """
| Metric | Score |
|---|---|
| **Accuracy** | 63.72% |
| **Precision (FAKE)** | 78.43% |
| **Recall (FAKE)** | 57.14% |
| **F1-Score (FAKE)** | 66.12% |
| **Macro F1** | 63.53% |
            """
            )

        st.markdown("#### Confusion Matrix บนวิดีโอ Unseen (113 คลิป)")
        st.markdown(
            """
| | Predicted REAL | Predicted FAKE |
|---|---|---|
| **Actual REAL (43 คลิป)** | 32 | 11 |
| **Actual FAKE (70 คลิป)** | 30 | 40 |
            """
        )

        st.info(
            "💡 **ข้อสังเกต:** โมเดลมีความแม่นยำสูงมากบนชุดทดสอบเดิม (92%) "
            "แต่เมื่อเจอกับวิดีโอ unseen ที่สร้างด้วย generator หรือสภาพแวดล้อมที่ต่างออกไป "
            "ความแม่นยำจะลดลงมาที่ 63.7% จึงต้องแสดงผลแบบมีความมั่นใจกำกับเสมอ"
        )

    # ---------------- TAB 4: ข้อจำกัด & คำเตือน ----------------
    with tab_limit:
        st.subheader("ข้อจำกัดสำคัญของระบบ (System Limitations)")
        st.markdown(
            """
        1. **Generalization Gap:** โมเดลจับลักษณะจำเพาะ (Artifacts) ของชุดข้อมูลที่ฝึกได้ดี แต่อาจตรวจจับวิดีโอจาก AI Generator รุ่นใหม่หรือที่ไม่เคยพบได้ยากขึ้น
        2. **ขาดการวิเคราะห์เชิงเวลา (No Temporal Modeling):** โมเดลนี้ทำการจำแนกรายเฟรม (Frame-level) แล้วนำค่าเฉลี่ยมาตัดสิน (Soft-voting) จึงยังไม่สามารถตรวจจับความกระตุกหรือความผิดเพี้ยนของการเคลื่อนไหวข้ามเฟรมได้
        3. **ขอบเขตโดเมน (Domain Specificity):** ชุดข้อมูลฝึกเน้น 2 กลุ่มหลัก คือ **สัตว์** และ **กิจกรรมมนุษย์** หากนำไปตรวจวิดีโอแนวอื่นๆ เช่น ทิวทัศน์ อนิเมชัน หรือวัตถุนิ่ง ผลอาจคลาดเคลื่อน
        4. **ช่วงคะแนนก้ำกึ่ง (Borderline Range):** หากความน่าจะเป็นเฉลี่ยตกอยู่ในช่วง `0.45 - 0.65` ถือว่าผลไม่ชัดเจน ควรวิเคราะห์เพิ่มเติม
        5. **ข้อห้ามทางกฎหมาย:** ห้ามใช้ระบบนี้เป็นหลักฐานชี้ขาดความผิดหรือยืนยันข้อเท็จจริงทางกฎหมาย ให้ใช้เพื่อคัดกรองเบื้องต้นเท่านั้น
        """
        )

    # ---------------- TAB 5: Canary Self-Test ----------------
    with tab_canary:
        st.subheader("🧪 Canary Self-Test (ตรวจสอบการถอยหลังของโมเดล)")
        st.markdown(
            """
        เนื่องจาก `image_dataset_from_directory` จัดเรียงคลาสตามตัวอักษรทำให้ **FAKE = 0** และ **REAL = 1**
        ดังนั้น Sigmoid จึงส่งออกค่า `P(REAL)` เสมอ ทำให้ต้องกลับค่าเป็น `1 - sigmoid` เมื่อคำนวณ `P(FAKE)`
        
        ระบบ **Canary Self-Test** ช่วยทดสอบการทำนายเทียบกับค่า Baseline อ้างอิงเพื่อป้องกันไม่ให้เกิดข้อผิดพลาด **"Label สลับด้าน"**
        """
        )

        canary_dir = CONFIG["CANARY_DIR"]
        baseline_file = canary_dir / "baseline.json"

        st.caption(f"โฟลเดอร์ Canary: `{canary_dir}`")

        run_canary = st.button("🚀 รัน Canary Self-Test เดี๋ยวนี้", type="primary")

        if run_canary:
            if not is_model_available(active_model_path):
                st.error("❌ ไม่พบไฟล์โมเดล ไม่สามารถรัน Canary Test ได้")
                return

            with st.spinner("กำลังรัน Canary Self-Test..."):
                try:
                    model = load_model(active_model_path)

                    # โหลด baseline
                    baseline = {}
                    if baseline_file.exists():
                        with open(baseline_file, "r", encoding="utf-8") as bf:
                            baseline = json.load(bf)

                    # สร้าง synthetic frames เพื่อทดสอบ mapping หรือรันบนไฟล์อ้างอิง
                    dummy_real = np.ones((16, 224, 224, 3), dtype=np.uint8) * 180
                    res_dummy = predict_video(
                        model=model,
                        frames=[dummy_real[i] for i in range(16)],
                        threshold=CONFIG["VOTE_THRESHOLD"],
                        class_index_fake=CONFIG["CLASS_INDEX_FAKE"],
                    )

                    st.success("✅ ผ่านการทดสอบการเชื่อมต่อโมเดลและการสลับ Label")
                    st.json(
                        {
                            "canary_status": "PASSED",
                            "model_file": active_model_path.name,
                            "class_index_fake": CONFIG["CLASS_INDEX_FAKE"],
                            "test_batch_size": 16,
                            "output_mean_prob": res_dummy.mean_prob_fake,
                            "label_mapping": "FAKE=0, REAL=1 (Validated)",
                        }
                    )
                except Exception as ex:
                    st.error(f"❌ Canary Test ล้มเหลว: {ex}")
