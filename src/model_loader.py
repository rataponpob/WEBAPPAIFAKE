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


def download_model_from_url(
    url: str,
    dest_path: Optional[Union[str, Path]] = None,
    progress_bar: Any = None,
    status_text: Any = None,
) -> Tuple[bool, str]:
    """ดาวน์โหลดโมเดลจาก URL (รองรับ GitHub Releases, Direct Link, Google Drive) พร้อม Progress bar และตรวจสอบความถูกต้อง"""
    import re
    import urllib.request

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # แปลงลิงก์ Google Drive ให้อยู่ในรูป direct download
    gdrive_match = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url)
    if gdrive_match:
        file_id = gdrive_match.group(1)
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0"

    if dest_path is None:
        clean_name = url.split("?")[0].split("/")[-1]
        if not (clean_name.endswith(".keras") or clean_name.endswith(".h5")):
            clean_name = "ResNet50_4dataset.keras"
        dest_path = models_dir / clean_name
    else:
        dest_path = Path(dest_path)

    temp_path = dest_path.with_suffix(dest_path.suffix + ".downloading")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f_out:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 512  # 512 KB

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    pct = min(1.0, downloaded / total_size)
                    mb_cur = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    if progress_bar is not None:
                        progress_bar.progress(pct, text=f"📥 กำลังดาวน์โหลด: {mb_cur:.1f} MB / {mb_tot:.1f} MB ({int(pct*100)}%)")
                    elif status_text is not None:
                        status_text.text(f"📥 กำลังดาวน์โหลด: {mb_cur:.1f} MB / {mb_tot:.1f} MB ({int(pct*100)}%)")

        if status_text is not None:
            status_text.text("🔍 กำลังตรวจสอบโครงสร้างไฟล์โมเดลด้วย TensorFlow...")

        # ทดสอบโหลดด้วย TensorFlow เพื่อตรวจสอบความสมบูรณ์
        tf.keras.models.load_model(str(temp_path))

        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)

        set_active_model_path(dest_path)
        return True, f"✅ ดาวน์โหลดและติดตั้งโมเดล `{dest_path.name}` สำเร็จ พร้อมใช้งาน!"

    except Exception as err:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        return False, f"❌ การดาวน์โหลดโมเดลล้มเหลว: {err}"

