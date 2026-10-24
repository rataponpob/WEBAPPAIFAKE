import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import tensorflow as tf
import streamlit as st
from src.config import CONFIG, PROJECT_ROOT


def is_model_available(model_path: Union[str, Path]) -> bool:
    """ตรวจสอบว่ามีไฟล์โมเดลอยู่จริงหรือไม่"""
    p = Path(model_path)
    return p.is_file() and p.stat().st_size > 0


def get_available_models() -> List[Path]:
    """ค้นหาไฟล์โมเดลทั้งหมด (.keras, .h5) ในโฟลเดอร์ models/"""
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        return []

    found = []
    for ext in ("*.keras", "*.h5"):
        found.extend(models_dir.glob(ext))

    # เรียงลำดับตามตัวอักษร
    return sorted(found, key=lambda x: x.name.lower())


def get_active_model_path() -> Path:
    """คืนค่าพาธของโมเดลที่กำลังใช้งานอยู่ ณ ปัจจุบัน"""
    # 1. ตรวจสอบใน session state ก่อน
    try:
        if "active_model_path" in st.session_state:
            p = Path(st.session_state["active_model_path"])
            if p.is_file():
                return p
    except Exception:
        pass

    # 2. ตรวจสอบใน CONFIG
    default_p = Path(CONFIG.get("MODEL_PATH", PROJECT_ROOT / "models" / "ResNet50_4dataset.keras"))
    if default_p.is_file():
        return default_p

    # 3. ตรวจสอบไฟล์โมเดลแรกที่มีอยู่ใน models/
    models = get_available_models()
    if models:
        return models[0]

    return default_p


def set_active_model_path(model_path: Union[str, Path]) -> None:
    """เปลี่ยนโมเดลที่กำลังใช้งาน"""
    p = Path(model_path)
    CONFIG["MODEL_PATH"] = p
    try:
        st.session_state["active_model_path"] = str(p)
    except Exception:
        pass


@st.cache_resource(show_spinner="กำลังโหลดโมเดลจากดิสก์…")
def load_model(model_path: Union[str, Path]) -> Any:
    """โหลดโมเดล Keras พร้อมแคชใน Streamlit process เพื่อความเร็ว"""
    p = Path(model_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดลที่: {p}\n"
            f"กรุณาวางไฟล์ '{p.name}' ลงในโฟลเดอร์ 'models/'"
        )

    model = tf.keras.models.load_model(str(p))
    return model


def get_model_info(model: Any) -> Dict[str, Any]:
    """ดึงข้อมูลโครงสร้างและจำนวนพารามิเตอร์ของโมเดล"""
    if model is None:
        return {}

    try:
        input_shape = getattr(model, "input_shape", None)
        total_params = getattr(model, "count_params", lambda: 0)()
        layers_count = len(getattr(model, "layers", []))
        return {
            "name": getattr(model, "name", "ResNet50_custom"),
            "input_shape": str(input_shape),
            "total_params": f"{total_params:,}",
            "layers_count": layers_count,
        }
    except Exception as e:
        return {"error": str(e)}


def save_uploaded_model(
    uploaded_file,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """บันทึกไฟล์โมเดลที่อัปโหลดลงโฟลเดอร์ models/ พร้อมตรวจสอบความถูกต้องด้วย TensorFlow"""
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    dest_path = models_dir / uploaded_file.name

    try:
        # บันทึกไฟล์ลงดิสก์
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # ทดสอบโหลดด้วย TensorFlow เพื่อยืนยันว่าเป็นโมเดล Keras ที่สมบูรณ์
        test_model = tf.keras.models.load_model(str(dest_path))
        info = get_model_info(test_model)

        # สลับมาใช้โมเดลใหม่ทันที
        set_active_model_path(dest_path)
        return True, f"✅ อัปโหลดและตรวจสอบโมเดล `{uploaded_file.name}` สำเร็จ พร้อมใช้งาน!", info

    except Exception as err:
        # หากไฟล์เสียหายหรือไม่ถูกต้อง ให้ลบทิ้งทันทีเพื่อไม่ให้ค้างในระบบ
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
        return False, f"❌ ไฟล์โมเดลไม่ถูกต้องหรือโหลดไม่สำเร็จ: {err}", None


def delete_model(model_path: Union[str, Path]) -> Tuple[bool, str]:
    """ลบไฟล์โมเดลออกจากโฟลเดอร์ models/"""
    p = Path(model_path)
    if not p.is_file():
        return False, f"ไม่พบไฟล์โมเดล: {p.name}"

    try:
        # ล้างแคชโหลดโมเดลของ streamlit
        try:
            load_model.clear()
        except Exception:
            pass

        p.unlink()

        # หากโมเดลที่ลบเป็นตัวที่ใช้งานอยู่ ให้สลับไปใช้โมเดลอื่นที่มี
        remaining = get_available_models()
        if remaining:
            set_active_model_path(remaining[0])
        else:
            set_active_model_path(PROJECT_ROOT / "models" / "ResNet50_4dataset.keras")

        return True, f"🗑️ ลบไฟล์โมเดล `{p.name}` เรียบร้อยแล้ว"
    except Exception as e:
        return False, f"ไม่สามารถลบไฟล์โมเดลได้: {e}"
