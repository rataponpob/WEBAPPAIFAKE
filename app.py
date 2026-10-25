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


def render_sidebar():
    st.sidebar.title("🎥 AI Video Detector")
    st.sidebar.caption("ระบบตรวจจับวิดีโอสร้างโดย AI ด้วย ResNet50")
    st.sidebar.markdown("---")

    # 1. เมนูหลัก (Navigation) - วางไว้บนสุด เด่นชัด ใช้งานสะดวก
    st.sidebar.subheader("🧭 เมนูหลัก (Navigation)")
    pages = {
        "🎯 ตรวจจับวิดีโอเดี่ยว": render_detect_page,
        "📦 ตรวจจับแบบกลุ่ม (Batch)": render_batch_page,
        "📜 ประวัติการตรวจ (History)": render_history_page,
        "ℹ️ ข้อมูลโมเดล (Model Info)": render_model_info_page,
    }

    selected_page_name = st.sidebar.radio(
        "เลือกหน้าเมนู:",
        options=list(pages.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # 2. สถานะและการเลือกโมเดล (GitHub Release)
    st.sidebar.subheader("🤖 สถานะโมเดล (GitHub Release)")

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

    # 3. การตั้งค่าการประมวลผล (Inference Settings)
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ การตั้งค่า (Settings)")

    n_frames = st.sidebar.slider(
        "จำนวนเฟรมที่สุ่มตรวจ (N_FRAMES):",
        min_value=4,
        max_value=100,
        value=st.session_state.get("cfg_n_frames", CONFIG["N_FRAMES"]),
        step=2,
        help="จำนวนเฟรมต่อคลิปที่ตัดมาสุ่มตรวจ ยิ่งมากยิ่งเสถียรแต่ใช้เวลาเพิ่มขึ้น (ปรับได้สูงสุด 100 เฟรม)",
    )
    st.session_state["cfg_n_frames"] = n_frames

    threshold = st.sidebar.slider(
        "เกณฑ์ตัดสิน FAKE (VOTE_THRESHOLD):",
        min_value=0.00,
        max_value=1.00,
        value=st.session_state.get(
            "cfg_vote_threshold", CONFIG["VOTE_THRESHOLD"]
        ),
        step=0.01,
        format="%.2f",
        help="ถ้า prob(FAKE) เฉลี่ย >= ค่านี้ ระบบจะตัดสินว่าเป็น FAKE (ปรับได้ตั้งแต่ 0.00 ถึง 1.00)",
    )
    st.session_state["cfg_vote_threshold"] = threshold

    st.sidebar.markdown("---")
    st.sidebar.caption("AI vs Real Video Detector v1.0 • Antigravity")

    return pages[selected_page_name]


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

        /* ปรับแต่ง Sidebar Navigation ให้ดูเป็นปุ่มเมนูสวยงาม */
        [data-testid="stSidebar"] [data-testid="stRadio"] > div {
            gap: 4px !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            margin-bottom: 2px !important;
            transition: all 0.18s ease-in-out !important;
            cursor: pointer !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            border-color: #3B82F6 !important;
            background: #F0F7FF !important;
            transform: translateX(3px) !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: #2563EB !important;
            border-color: #1D4ED8 !important;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
            display: none !important;
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
