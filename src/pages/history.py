import json
import time
import pandas as pd
import streamlit as st


def render_history_page():
    st.header("📜 ประวัติการตรวจสอบในเซสชัน (Session History)")
    st.markdown(
        "บันทึกประวัติการวิเคราะห์วิดีโอทั้งหมดที่ทำในเซสชันปัจจุบัน "
        "(ระบบ stateless ไม่บันทึกข้อมูลค้างไว้บนเซิร์ฟเวอร์หลังปิดเบราว์เซอร์)"
    )

    history = st.session_state.get("history", [])

    if not history:
        st.info("ℹ️ ยังไม่มีประวัติการตรวจสอบในเซสชันนี้ ลองไปตรวจวิดีโอในหน้าแรกก่อน")
        return

    df = pd.DataFrame(history)

    # ตัวกรอง Filter
    col1, col2 = st.columns([2, 2])
    with col1:
        filter_verdict = st.selectbox(
            "กรองตามผลการตรวจ:",
            options=["ทั้งหมด (All)", "FAKE", "REAL", "เฉพาะ Borderline"],
        )

    filtered_df = df.copy()
    if filter_verdict == "FAKE":
        filtered_df = filtered_df[filtered_df["verdict"] == "FAKE"]
    elif filter_verdict == "REAL":
        filtered_df = filtered_df[filtered_df["verdict"] == "REAL"]
    elif filter_verdict == "เฉพาะ Borderline":
        filtered_df = filtered_df[filtered_df["borderline"] == True]

    # สถิติภาพรวม
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("จำนวนคลิปที่ตรวจ", f"{len(filtered_df)} ไฟล์")
    n_fake = sum(filtered_df["verdict"] == "FAKE")
    n_real = sum(filtered_df["verdict"] == "REAL")
    c2.metric("ตรวจพบ FAKE", f"{n_fake} ไฟล์")
    c3.metric("ตรวจพบ REAL", f"{n_real} ไฟล์")
    n_border = sum(filtered_df.get("borderline", [False] * len(filtered_df)))
    c4.metric("ก้ำกึ่ง (Borderline)", f"{n_border} ไฟล์")

    # ตารางแสดงผล
    st.dataframe(
        filtered_df.style.map(
            lambda v: "background-color: #FEE2E2; font-weight: bold;"
            if v == "FAKE"
            else (
                "background-color: #DCFCE7; font-weight: bold;"
                if v == "REAL"
                else ""
            ),
            subset=["verdict"],
        ),
        use_container_width=True,
    )

    st.divider()

    # ปุ่มจัดการประวัติ
    col_dl_csv, col_dl_json, col_clear = st.columns([2, 2, 2])

    with col_dl_csv:
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ ส่งออกประวัติเป็น CSV",
            data=csv_bytes,
            file_name=f"history_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl_json:
        json_str = json.dumps(history, ensure_ascii=False, indent=2)
        st.download_button(
            label="⬇️ ส่งออกประวัติเป็น JSON",
            data=json_str,
            file_name=f"history_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_clear:
        if st.button("🗑️ ล้างประวัติทั้งหมด", type="secondary", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()
