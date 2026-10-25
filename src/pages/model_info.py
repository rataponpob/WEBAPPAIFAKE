import time
from pathlib import Path
import streamlit as st

from src.config import CONFIG
from src.model_loader import (
    download_model_from_url,
    get_active_model_path,
    get_available_models,
    get_model_info,
    is_model_available,
    load_model,
    set_active_model_path,
)


def render_model_info_page():
    st.header("⚙️ จัดการโมเดล (Model Management)")
    st.caption("ตรวจสอบ สลับใช้งาน และติดตั้งโมเดลจาก GitHub Release")

    active_model_path = get_active_model_path()
    available_models = get_available_models()

    # การ์ดสถานะโมเดลปัจจุบัน
    st.markdown("##### 🤖 โมเดลที่ใช้งานอยู่ ณ ปัจจุบัน")
    if is_model_available(active_model_path):
        size_mb = active_model_path.stat().st_size / (1024 * 1024)
        c_status1, c_status2, c_status3 = st.columns(3)
        with c_status1:
            st.metric("ชื่อโมเดล", active_model_path.name)
        with c_status2:
            st.metric("ขนาดไฟล์", f"{size_mb:.1f} MB")
        with c_status3:
            st.metric("สถานะ", "🟢 พร้อมใช้งาน")
    else:
        st.error("🔴 ยังไม่มีโมเดลพร้อมใช้งานในเซิร์ฟเวอร์ กรุณาดาวน์โหลดจาก GitHub Release ด้านล่าง")

    st.markdown("---")

    # ส่วนที่ 1: รายการโมเดลในระบบ
    st.subheader("📁 รายการโมเดลในระบบ")
    if not available_models:
        st.warning("⚠️ ยังไม่พบไฟล์โมเดลในโฟลเดอร์ `models/`")
    else:
        for m_path in available_models:
            is_active = (m_path.name == active_model_path.name)
            m_size_mb = m_path.stat().st_size / (1024 * 1024)

            c_name, c_size, c_stat, c_act = st.columns([5, 2, 2, 2])
            with c_name:
                st.markdown(f"**`{m_path.name}`**")
            with c_size:
                st.caption(f"{m_size_mb:.1f} MB")
            with c_stat:
                if is_active:
                    st.success("🟢 ใช้งานอยู่")
                else:
                    st.caption("⚪ สแตนด์บาย")
            with c_act:
                if not is_active:
                    if st.button("เลือกใช้", key=f"use_{m_path.name}", use_container_width=True):
                        set_active_model_path(m_path)
                        st.rerun()
            st.divider()

    # ส่วนที่ 2: ดาวน์โหลดจาก GitHub Release
    st.subheader("📦 ติดตั้งโมเดลจาก GitHub Release")
    st.caption("ดาวน์โหลดไฟล์โมเดล Keras (.keras) จาก GitHub Release Asset ตรงเข้าสู่ระบบ")

    rel_url = st.text_input(
        "GitHub Release Asset URL:",
        value=CONFIG.get("DEFAULT_MODEL_URL", ""),
        placeholder="https://github.com/.../releases/download/.../model.keras",
        key="model_info_gh_url",
        help="ระบุ URL ของไฟล์โมเดล .keras จาก GitHub Release",
    )

    if st.button("📥 ดาวน์โหลดและติดตั้งโมเดล", type="primary", key="btn_model_info_dl"):
        if rel_url.strip():
            pbar = st.progress(0, text="กำลังเตรียมการดาวน์โหลด...")
            s_txt = st.empty()
            with st.spinner("กำลังดาวน์โหลดและตรวจสอบโมเดล..."):
                ok, msg = download_model_from_url(rel_url.strip(), progress_bar=pbar, status_text=s_txt)
                if ok:
                    st.success(msg)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.warning("กรุณาระบุ URL ก่อนดาวน์โหลด")
