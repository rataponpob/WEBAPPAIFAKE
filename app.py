import os
# ปิดคำเตือน oneDNN และแก้ไข Windows DLL loading conflict (Error 1114 DllMain False)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# โหลด TensorFlow ก่อนไลบรารี C-extensions อื่นๆ (เช่น pandas/pyarrow)
import tensorflow as tf

import streamlit as st

st.set_page_config(
    page_title="AI vs Real Video Detector",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import CONFIG
from src.model_loader import (
    download_model_from_url,
    get_active_model_path,
    get_available_models,
    is_model_available,
    set_active_model_path,
)
from src.pages.batch import render_batch_page
from src.pages.detect import render_detect_page
from src.pages.history import render_history_page
from src.pages.model_info import render_model_info_page


PAGES = {
    "detect": {"label": "🎯 ตรวจจับวิดีโอเดี่ยว", "func": render_detect_page},
    "batch": {"label": "📦 ตรวจจับแบบกลุ่ม (Batch)", "func": render_batch_page},
    "history": {"label": "📜 ประวัติการตรวจ (History)", "func": render_history_page},
    "model": {"label": "⚙️ จัดการโมเดล (Manage)", "func": render_model_info_page},
}


def render_sidebar():
    st.sidebar.markdown(
        """
        <div style="padding: 2px 0 6px 0;">
            <h2 style="margin: 0; font-size: 1.3rem; color: #1E293B; font-weight: 700;">🎥 AI Video Detector</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # 1. แถบนำทาง (Navbar) สไตล์โมเดิร์น (ไม่ใช่ Option Radio)
    st.sidebar.markdown("<p style='font-size: 0.76rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px;'>🧭 เมนูหลัก (NAVBAR)</p>", unsafe_allow_html=True)

    if "current_page_key" not in st.session_state:
        st.session_state["current_page_key"] = "detect"

    for key, pinfo in PAGES.items():
        is_current = (st.session_state["current_page_key"] == key)
        btn_type = "primary" if is_current else "secondary"
        if st.sidebar.button(
            pinfo["label"],
            key=f"nav_btn_{key}",
            type=btn_type,
            use_container_width=True,
        ):
            if st.session_state["current_page_key"] != key:
                st.session_state["current_page_key"] = key
                st.rerun()

    st.sidebar.markdown("---")

    # 2. สถานะและการเลือกโมเดล (GitHub Release)
    st.sidebar.markdown("<p style='font-size: 0.76rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px;'>🤖 สถานะโมเดล</p>", unsafe_allow_html=True)

    available_models = get_available_models()
    active_model_path = get_active_model_path()

    if available_models:
        model_names = [m.name for m in available_models]
        cur_idx = (
            model_names.index(active_model_path.name)
            if active_model_path.name in model_names
            else 0
        )
        selected_name = st.sidebar.selectbox(
            "โมเดลที่เลือกใช้งาน:",
            options=model_names,
            index=cur_idx,
            label_visibility="collapsed",
            help="สามารถเลือกสลับโมเดลที่ติดตั้งไว้แล้วในระบบได้ทันที",
        )
        selected_path = next(
            (m for m in available_models if m.name == selected_name),
            active_model_path,
        )
        if selected_path != active_model_path:
            set_active_model_path(selected_path)
            st.rerun()

        st.sidebar.success(f"🟢 พร้อมใช้งาน (`{active_model_path.name}`)")
    else:
        st.sidebar.error("🔴 ยังไม่พบไฟล์โมเดลในเซิร์ฟเวอร์")
        default_dl_url = CONFIG.get("DEFAULT_MODEL_URL", "")
        if default_dl_url:
            st.sidebar.info("💡 สามารถกดดาวน์โหลดโมเดลจาก GitHub Release ได้ทันที")
            if st.sidebar.button("📥 ดาวน์โหลดโมเดลจาก GitHub Release", type="primary", use_container_width=True, key="btn_auto_dl_sidebar"):
                pbar = st.sidebar.progress(0, text="กำลังเตรียมการดาวน์โหลด...")
                st_status = st.sidebar.empty()
                ok, msg = download_model_from_url(default_dl_url, progress_bar=pbar, status_text=st_status)
                if ok:
                    st.sidebar.success(msg)
                    time_mod = __import__("time")
                    time_mod.sleep(1)
                    st.rerun()
                else:
                    st.sidebar.error(msg)

    # ตัวเลือกดาวน์โหลดโมเดลจาก GitHub Release
    with st.sidebar.expander("📦 ติดตั้งโมเดลจาก GitHub Release", expanded=not available_models):
        st.caption("ดาวน์โหลดโมเดล (.keras) จาก GitHub Release Asset URL")
        input_url = st.text_input(
            "GitHub Release URL:",
            value=CONFIG.get("DEFAULT_MODEL_URL", "") if not available_models else "",
            placeholder="https://github.com/.../releases/download/.../model.keras",
            key="input_model_url",
            help="ระบุ URL ของไฟล์โมเดล .keras จาก GitHub Release",
        )
        if st.button("📥 ดาวน์โหลดและติดตั้งโมเดล", key="btn_start_dl_url", use_container_width=True):
            if input_url.strip():
                pbar2 = st.progress(0, text="กำลังเตรียมการดาวน์โหลด...")
                st_stat2 = st.empty()
                with st.spinner("กำลังดาวน์โหลดและตรวจสอบโครงสร้างโมเดล..."):
                    ok, msg = download_model_from_url(input_url.strip(), progress_bar=pbar2, status_text=st_stat2)
                    if ok:
                        st.sidebar.success(msg)
                        time_mod = __import__("time")
                        time_mod.sleep(1)
                        st.rerun()
                    else:
                        st.sidebar.error(msg)
            else:
                st.sidebar.warning("กรุณาระบุ URL ก่อนเริ่มดาวน์โหลด")

    st.sidebar.markdown("---")
    st.sidebar.caption("AI vs Real Video Detector v1.0 • Antigravity")

    current_key = st.session_state.get("current_page_key", "detect")
    return PAGES.get(current_key, PAGES["detect"])["func"]


def inject_compact_styles():
    """ปรับแต่ง CSS ให้ UI มีขนาดกะทัดรัด พอดีสายตา ไม่ใหญ่เทอะทะ"""
    st.markdown(
        """
        <style>
        /* บีบความกว้าง Layout ให้พอดีสายตา */
        .main .block-container {
            max-width: 1060px !important;
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            margin: 0 auto !important;
        }

        /* ตกแต่ง Sidebar Navbar Buttons ให้ดูพรีเมียม สไตล์แท็บโมเดิร์น */
        [data-testid="stSidebar"] button[kind="primary"] {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 600 !important;
            box-shadow: 0 3px 8px rgba(37, 99, 235, 0.28) !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 0.55rem 0.95rem !important;
            border-radius: 8px !important;
            font-size: 0.88rem !important;
        }

        [data-testid="stSidebar"] button[kind="secondary"] {
            background: #FFFFFF !important;
            color: #334155 !important;
            border: 1px solid #E2E8F0 !important;
            font-weight: 500 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 0.55rem 0.95rem !important;
            border-radius: 8px !important;
            font-size: 0.88rem !important;
            transition: all 0.18s ease-in-out !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        }

        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            border-color: #3B82F6 !important;
            color: #2563EB !important;
            background: #EFF6FF !important;
            transform: translateX(3px) !important;
        }

        /* ปรับขนาดหัวข้อและข้อความ */
        h1 { font-size: 1.55rem !important; font-weight: 700 !important; margin-bottom: 0.4rem !important; }
        h2 { font-size: 1.25rem !important; font-weight: 700 !important; margin-bottom: 0.35rem !important; }
        h3 { font-size: 1.05rem !important; font-weight: 600 !important; margin-bottom: 0.3rem !important; }
        h4, h5, h6 { font-size: 0.92rem !important; font-weight: 600 !important; margin-bottom: 0.25rem !important; }
        p, span, label, div { font-size: 0.88rem; }

        /* ควบคุมขนาด Video Player ให้พอดีขอบตาม Aspect Ratio จริง */
        [data-testid="stVideo"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
            background: transparent !important;
        }
        [data-testid="stVideo"] video, video {
            display: block !important;
            max-width: 100% !important;
            width: auto !important;
            height: auto !important;
            max-height: 280px !important;
            object-fit: contain !important;
            border-radius: 8px !important;
            background: #0B0F19 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        }

        /* Metrics ขนาดกะทัดรัด */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.76rem !important;
        }

        /* ปรับขนาด Tab และปุ่ม */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 5px 12px !important;
            font-size: 0.84rem !important;
        }
        button[kind="primary"], button[kind="secondary"] {
            padding: 0.32rem 0.85rem !important;
            font-size: 0.86rem !important;
            border-radius: 6px !important;
        }

        /* ลดช่องว่างส่วนเกิน */
        .element-container {
            margin-bottom: 0.3rem !important;
        }
        div[data-testid="stVerticalBlock"] > div {
            gap: 0.45rem !important;
        }
        hr {
            margin-top: 0.7rem !important;
            margin-bottom: 0.7rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_compact_styles()
    page_func = render_sidebar()
    page_func()


if __name__ == "__main__":
    main()

