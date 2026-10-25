from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _resolve_model_path() -> Path:
    """ค้นหาไฟล์โมเดลที่พร้อมใช้งานใน models/"""
    for filename in ("ResNet50_4dataset.keras", "ResNet50_best_image.keras"):
        p = PROJECT_ROOT / "models" / filename
        if p.is_file():
            return p
    return PROJECT_ROOT / "models" / "ResNet50_4dataset.keras"


CONFIG = {
    # --- โมเดล ---
    "MODEL_PATH": _resolve_model_path(),
    "MODEL_NAME": "ResNet50",
    "IMG_SIZE": (224, 224),
    "CLASS_INDEX_FAKE": 1,          # FAKE=1, REAL=0 (สลับฝั่งตามโมเดล)
    # --- inference (ค่า default ตาม canary ที่ผ่านแล้วในโน้ตบุ๊ก) ---
    "N_FRAMES": 16,
    "VOTE_THRESHOLD": 0.55,
    "VIDEO_EXTS": (".mp4", ".mov", ".avi", ".webm", ".mkv"),
    "MAX_UPLOAD_MB": 500,
    "BORDERLINE_BAND": (0.45, 0.65),  # ช่วงที่ถือว่าผลไม่ชัด
    "CANARY_DIR": PROJECT_ROOT / "assets" / "canary",
    "DEFAULT_MODEL_URL": "https://github.com/rataponpob/WEBAPPAIFAKE/releases/download/v1.0.0/ResNet50_4dataset.keras",
}

MODEL_REGISTRY = {
    "ResNet50": {
        "path": CONFIG["MODEL_PATH"],
        "preprocess": "tensorflow.keras.applications.resnet50.preprocess_input",
    },
}
