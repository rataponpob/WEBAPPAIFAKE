# 🎥 AI vs Real Video Detector (Streamlit Web Application)

เว็บแอปพลิเคชันสำหรับตรวจจับวิดีโอที่สร้างโดย AI (Sora และ generative video models อื่นๆ) เทียบกับวิดีโอที่ถ่ายจริง โดยใช้โมเดล Deep Learning **ResNet50 (Transfer Learning)** ตามการออกแบบใน `streamlit_app_design.md`

---

## 🌟 คุณสมบัติเด่น (Features)

1. **🎯 Single Video Detection:**
   - อัปโหลดวิดีโอเดี่ยว (`.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`) พร้อมเล่นพรีวิว
   - ประมวลผลแบบ Uniform Frame Sampling 16 เฟรมตลอดคลิป
   - แสดงผลชัดเจนด้วย **Verdict Card**, **Plotly Gauge** (0–100% P(FAKE)), **Frame-by-Frame Bar Chart** และ **Frame Gallery**
   - รองรับการดาวน์โหลดรายงานสรุปเป็นไฟล์ JSON

2. **📦 Batch Video Detection:**
   - อัปโหลดหลายไฟล์พร้อมกันเพื่อวิเคราะห์แบบชุด
   - แสดง Progress Bar ติดตามสถานะแบบ Real-time
   - สรุปตัวชี้วัด Accuracy และ Confusion Matrix อัตโนมัติหากชื่อไฟล์มีระบุ Ground Truth (`real`/`fake`)
   - ส่งออกผลลัพธ์เป็นตาราง CSV (`predictions.csv`)

3. **📜 Session History:**
   - เก็บบันทึกประวัติการตรวจในเซสชันปัจจุบัน (Stateless — ไม่เก็บค้างบนเซิร์ฟเวอร์)
   - กรองผลลัพธ์ตามคำตัดสิน (FAKE / REAL / Borderline)
   - ส่งออกประวัติทั้งหมดเป็น CSV หรือ JSON ได้ทันที

4. **ℹ️ Model Information & Canary Self-Test:**
   - แสดงรายละเอียดโครงสร้าง ResNet50 Backbone + Classification Head
   - ตารางเปรียบเทียบตัวชี้วัดระหว่าง In-Distribution (Test Set F1 92.3%) กับ Out-of-Distribution (Unseen Set Accuracy 63.7%)
   - **Canary Self-Test:** ปุ่มทดสอบตรวจสอบการถอยหลังของระบบ ป้องกันบัก Label สลับด้าน (`FAKE=0, REAL=1`)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
ProjectAIREAL/
├── app.py                      # จุดเริ่มต้นแอป: Navigation + Sidebar + Routing
├── requirements.txt            # รายการ Dependencies
├── streamlit_app_design.md      # เอกสารออกแบบระบบฉบับสมบูรณ์
├── .streamlit/
│   └── config.toml             # ตั้งค่าขีดจำกัดขนาดไฟล์ 500MB และ Theme
├── models/
│   ├── README.md               # คำแนะนำการวางไฟล์โมเดล
│   └── ResNet50_best_image.keras (วางไฟล์ที่นี่)
├── assets/
│   └── canary/
│       └── baseline.json       # ค่าอ้างอิงสำหรับ Canary Regression Test
├── src/
│   ├── __init__.py
│   ├── config.py               # ค่าคงที่และคอนฟิกูเรชันหลัก
│   ├── model_loader.py         # ตัวโหลดโมเดลพร้อม @st.cache_resource
│   ├── frame_extractor.py      # ฟังก์ชันสุ่มตัดเฟรม (Uniform Sampling)
│   ├── predictor.py            # การทำนายและแปลง Label (Sigmoid -> P(FAKE))
│   ├── visuals.py              # กราฟิก Plotly, Verdict Cards, Frame Gallery
│   └── pages/
│       ├── detect.py           # หน้าตรวจวิดีโอเดี่ยว
│       ├── batch.py            # หน้าตรวจหลายไฟล์พร้อมกัน
│       ├── history.py          # หน้าประวัติการตรวจในเซสชัน
│       └── model_info.py       # หน้าข้อมูลโมเดลและ Canary Self-Test
└── tests/
    ├── test_frame_extractor.py # ทดสอบสูตรคำนวณ Uniform Sampling
    ├── test_predictor.py       # ทดสอบการแปลง Label และ Borderline Band
    └── test_canary.py          # ทดสอบ Canary Baseline Schema
```

---

## 🚀 วิธีติดตั้งและเริ่มต้นใช้งาน (Getting Started)

### 1. ติดตั้ง Python และ Dependencies

แนะนำให้ใช้ Python 3.11 หรือ 3.12:

```bash
# ติดตั้ง Virtual Environment
python -m venv .venv

# เปิดใช้งาน venv (บน Windows)
.venv\Scripts\activate

# ติดตั้งแพ็กเกจ
pip install -r requirements.txt
```

*(หรือหากใช้ `uv`: `uv venv --python 3.11 .venv` แล้ว `uv pip install -r requirements.txt`)*

### 2. วางไฟล์โมเดล

นำไฟล์โมเดล `ResNet50_best_image.keras` ที่ได้จากการฝึกในโน้ตบุ๊กบน Google Colab / Drive มาวางที่โฟลเดอร์:
```text
models/ResNet50_best_image.keras
```

### 3. รันเว็บแอป Streamlit

```bash
streamlit run app.py
```
เปิดเบราว์เซอร์ไปที่ `http://localhost:8501`

---

## 🧪 การทดสอบ Unit Tests

ทดสอบความถูกต้องของโมดูล Frame Extractor, Predictor Label Mapping และ Canary Baseline:

```bash
pytest
```

---

## ⚠️ ข้อจำกัดสำคัญ (Important Disclaimers)

1. **Generalization Gap:** ตัวเลข 92.06% F1 มาจาก Test Set ที่อยู่ในชุดข้อมูลเดียวกับชุดฝึก; บนวิดีโอ unseen ความแม่นยำระดับวิดีโออยู่ที่ **63.7%**
2. **Frame-Level Classifier:** โมเดลประเมินรายเฟรมด้วย Soft-voting ยังไม่ได้ตรวจจับความต่อเนื่องเชิงเวลา (Temporal artifacts)
3. **Domain Constraints:** ฝึกบน 2 โดเมนหลักคือ "สัตว์" และ "กิจกรรมมนุษย์"
4. **Borderline:** คะแนนในช่วง `0.45 - 0.65` ควรถือเป็น "ไม่แน่นอน"
5. **ห้ามใช้เป็นหลักฐานเด็ดขาดทางกฎหมาย**
