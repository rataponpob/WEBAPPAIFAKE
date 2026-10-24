from typing import List, Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from src.predictor import PredictionResult


def render_verdict_card(result: PredictionResult):
    """เรนเดอร์การ์ดผลลัพธ์คำตัดสินแบบสวยงาม พรีเมียม และชัดเจน"""
    is_fake = result.label.lower() == "fake"
    is_borderline = result.borderline

    if is_borderline:
        card_bg = "linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%)"
        border_color = "#F59E0B"
        badge_bg = "#F59E0B"
        badge_text = "#FFFFFF"
        title = "⚠️ ไม่แน่นอน (BORDERLINE)"
        desc = f"ค่าความน่าจะเป็นเฉลี่ย ({result.mean_prob_fake:.1%}) อยู่ใกล้เส้นแบ่งเกณฑ์ ({result.threshold:.2f}) ควรตรวจสอบคลิปเพิ่มเติมหรือพิจารณาบริบทอื่นร่วมด้วย"
    elif is_fake:
        card_bg = "linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%)"
        border_color = "#EF4444"
        badge_bg = "#EF4444"
        badge_text = "#FFFFFF"
        title = "🔴 ผลการตรวจ: วิดีโอสร้างโดย AI (FAKE)"
        desc = f"โมเดลประเมินว่ามีลักษณะหรือ Artifacts สอดคล้องกับวิดีโอสังเคราะห์จาก AI (โอกาส FAKE: {result.mean_prob_fake:.1%})"
    else:
        card_bg = "linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%)"
        border_color = "#10B981"
        badge_bg = "#10B981"
        badge_text = "#FFFFFF"
        title = "🟢 ผลการตรวจ: วิดีโอถ่ายจริง (REAL)"
        desc = f"โมเดลประเมินว่ามีลักษณะสอดคล้องกับวิดีโอถ่ายจริงตามธรรมชาติ (โอกาส REAL: {(1.0 - result.mean_prob_fake):.1%})"

    card_html = f"""
    <div style="
        background: {card_bg};
        border-left: 5px solid {border_color};
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <h3 style="margin: 0 0 4px 0; color: #1E293B; font-size: 1.1rem; font-weight: 700;">{title}</h3>
                <p style="margin: 0; color: #475569; font-size: 0.88rem;">{desc}</p>
            </div>
            <div style="
                background: {badge_bg};
                color: {badge_text};
                padding: 6px 14px;
                border-radius: 16px;
                font-weight: 700;
                font-size: 0.95rem;
                letter-spacing: 0.3px;
            ">
                {result.mean_prob_fake:.1%} P(FAKE)
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "คำตัดสิน (Verdict)",
            result.label.upper(),
            delta="FAKE" if is_fake else "REAL",
            delta_color="inverse" if is_fake else "normal",
        )
    with col2:
        st.metric("โอกาสเป็น FAKE", f"{result.mean_prob_fake:.1%}")
    with col3:
        st.metric("ระดับความมั่นใจ", result.confidence_level)
    with col4:
        st.metric(
            "เฟรมที่วิเคราะห์",
            f"{result.n_frames_used} เฟรม",
            f"เกณฑ์: {result.threshold:.2f}",
        )


def plot_gauge(mean_prob: float, threshold: float = 0.55) -> go.Figure:
    """เกจแสดงระดับความน่าจะเป็น FAKE พร้อมแถบสีและตำแหน่งเกณฑ์"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=mean_prob * 100,
            number={"suffix": "%", "font": {"size": 26, "color": "#0F172A"}},
            title={
                "text": "ความน่าจะเป็น AI / FAKE (%)",
                "font": {"size": 13, "color": "#475569"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#94A3B8",
                },
                "bar": {"color": "#1E293B", "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 45], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [45, 65], "color": "rgba(245, 158, 11, 0.3)"},
                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "#DC2626", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif"),
    )
    return fig


def plot_frame_probs(
    frame_probs: List[float], threshold: float = 0.55
) -> go.Figure:
    """กราฟแท่งความน่าจะเป็น Fake รายเฟรม พร้อมเส้นเกณฑ์ Threshold"""
    n = len(frame_probs)
    frame_numbers = [f"F{i+1}" for i in range(n)]

    colors = [
        "#EF4444" if p >= threshold else "#10B981" for p in frame_probs
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=frame_numbers,
            y=frame_probs,
            marker_color=colors,
            text=[f"{p:.0%}" for p in frame_probs],
            textposition="auto",
            name="P(FAKE)",
            hovertemplate="%{x}<br>โอกาส FAKE: %{y:.2%}<extra></extra>",
        )
    )

    # Threshold horizontal line
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=n - 0.5,
        y0=threshold,
        y1=threshold,
        line=dict(color="#DC2626", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=n - 1,
        y=threshold,
        text=f"เกณฑ์ ({threshold:.2f})",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-20,
        font=dict(color="#DC2626", size=10),
    )

    fig.update_layout(
        title=dict(text="ความน่าจะเป็น FAKE แยกรายเฟรม", font=dict(size=13)),
        xaxis_title="ลำดับเฟรมสุ่ม",
        yaxis_title="P(FAKE)",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=220,
        margin=dict(l=15, r=15, t=35, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FAFC",
        font=dict(family="sans-serif", size=11),
    )
    return fig


def render_frame_gallery(
    frames: List[np.ndarray],
    frame_probs: List[float],
    max_display: int = 100,
    threshold: float = 0.55,
):
    """แสดงแกลเลอรีภาพเฟรมที่สุ่มมาวิเคราะห์ ขนาดกะทัดรัด แถวละ 6 ภาพ"""
    st.markdown("##### 🖼️ ภาพเฟรมที่ถูกสุ่มตรวจ (Sampled Frames)")

    n_frames = min(len(frames), max_display)
    if n_frames == 0:
        st.info("ไม่มีภาพเฟรมแสดง")
        return

    # แสดงแถวละ 6 เฟรมเพื่อให้กะทัดรัด ไม่ใหญ่เทอะทะ
    cols_per_row = 6
    for row_start in range(0, n_frames, cols_per_row):
        cols = st.columns(cols_per_row)
        for c_idx in range(cols_per_row):
            idx = row_start + c_idx
            if idx < n_frames:
                frame = frames[idx]
                prob = frame_probs[idx] if idx < len(frame_probs) else 0.0
                tag_label = "🔴 FAKE" if prob >= threshold else "🟢 REAL"

                with cols[c_idx]:
                    st.image(
                        frame,
                        caption=f"F#{idx+1} | {tag_label} ({prob:.0%})",
                        use_container_width=True,
                    )
