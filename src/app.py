import os
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Flower Classification (UAP)",
    page_icon="🌸",
    layout="wide"
)

# =========================
# PATHS & CONFIG
# =========================
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODEL_ZOO = {
    "CNN Scratch": str(MODELS_DIR / "scratch_cnn.keras"),
    "EfficientNet-B0": str(MODELS_DIR / "effnetb0.keras"),
    "MobileNetV2": str(MODELS_DIR / "mobilenetv2.keras"),
}

CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
IMG_SIZE = (224, 224)

# =========================
# SESSION STATE (HISTORY)
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# CUSTOM CSS (PINK x BLACK)
# =========================
st.markdown(
    """
<style>
  :root{
    --bg: #070812;
    --panel: rgba(255,255,255,0.055);
    --panel2: rgba(255,255,255,0.035);
    --stroke: rgba(255,255,255,0.10);
    --text: #f3f4ff;
    --muted: rgba(243,244,255,0.72);

    --pink: #ff2ea6;
    --pink2: #ff5bc0;
    --pinkGlow: rgba(255,46,166,0.25);
    --pinkGlow2: rgba(255,46,166,0.12);
  }

  /* App background */
  .stApp{
    background:
      radial-gradient(900px 500px at 15% 10%, var(--pinkGlow), transparent 60%),
      radial-gradient(800px 450px at 85% 20%, var(--pinkGlow2), transparent 60%),
      radial-gradient(900px 500px at 50% 95%, rgba(255,46,166,0.10), transparent 65%),
      var(--bg);
    color: var(--text);
  }

  /* Sidebar */
  section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border-right: 1px solid rgba(255,46,166,0.20);
  }
  section[data-testid="stSidebar"] *{
    color: var(--text);
  }

  /* Containers spacing */
  .block-container{ padding-top: 1.1rem; }

  /* Hero */
  .hero{
    padding: 18px 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255,46,166,0.10), rgba(255,255,255,0.04));
    border: 1px solid rgba(255,46,166,0.22);
    box-shadow: 0 18px 55px rgba(0,0,0,0.40);
    margin-bottom: 16px;
  }
  .hero h1{
    margin: 0;
    font-size: 44px;
    letter-spacing: .2px;
  }
  .hero p{
    margin: 6px 0 0 0;
    color: var(--muted);
    font-size: 14px;
  }
  .hero .tag{
    display:inline-block;
    margin-top:10px;
    padding:7px 12px;
    border-radius:999px;
    background: rgba(255,46,166,0.12);
    border: 1px solid rgba(255,46,166,0.28);
    font-weight: 800;
    font-size: 13px;
  }

  /* Cards */
  .card{
    padding: 16px 16px;
    border-radius: 18px;
    background: var(--panel);
    border: 1px solid var(--stroke);
    box-shadow: 0 14px 45px rgba(0,0,0,0.30);
    transition: transform .12s ease, border-color .12s ease;
  }
  .card:hover{
    transform: translateY(-2px);
    border-color: rgba(255,46,166,0.32);
  }
  .card-title{
    font-size: 18px;
    font-weight: 900;
    margin: 0 0 8px 0;
  }
  .muted{ color: var(--muted); font-size: 13px; }

  /* Badges */
  .badgePink{
    display:inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,46,166,0.12);
    border: 1px solid rgba(255,46,166,0.28);
    font-weight: 900;
    font-size: 13px;
  }

  /* Buttons */
  div.stButton > button{
    background: linear-gradient(135deg, var(--pink), var(--pink2));
    color: #0a0b10 !important;
    border: 0 !important;
    border-radius: 12px !important;
    font-weight: 900 !important;
    box-shadow: 0 10px 26px rgba(255,46,166,0.25);
  }
  div.stButton > button:hover{
    filter: brightness(1.06);
    box-shadow: 0 12px 34px rgba(255,46,166,0.35);
  }

  /* Inputs */
  div[data-testid="stFileUploader"] section{
    border: 1px dashed rgba(255,46,166,0.35) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.02) !important;
  }

  /* Table + Dataframe */
  .stDataFrame, .stTable{
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.10);
  }

  /* Progress */
  div[data-testid="stProgressBar"] > div{
    background: rgba(255,46,166,0.20) !important;
  }
  div[data-testid="stProgressBar"] > div > div{
    background: linear-gradient(90deg, var(--pink), var(--pink2)) !important;
  }
</style>
    """,
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
@st.cache_resource
def load_keras_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    return tf.keras.models.load_model(path)

def letterbox(pil_img: Image.Image, size=(224, 224), bg_color=(0, 0, 0)):
    """Resize menjaga rasio (contain), lalu padding ke size."""
    img = pil_img.convert("RGB")
    img = ImageOps.contain(img, size)
    new_img = Image.new("RGB", size, bg_color)
    paste_x = (size[0] - img.size[0]) // 2
    paste_y = (size[1] - img.size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img

def preprocess_image(pil_img: Image.Image):
    img = letterbox(pil_img, IMG_SIZE, bg_color=(0, 0, 0))
    x = np.array(img).astype("float32")   # tetap 0..255 (tanpa normalisasi)
    x = np.expand_dims(x, axis=0)
    return x, img

def topk_from_proba(proba, k=3):
    idx = np.argsort(proba)[::-1][:k]
    return [(CLASS_NAMES[i], float(proba[i])) for i in idx]

# =========================
# HERO
# =========================
st.markdown(
    """
    <div class="hero">
      <h1>🌸 Flower Classification (UAP)</h1>
      <p>Upload gambar bunga → prediksi kelas + confidence + Top-3 probabilitas → tersimpan ke history.</p>
      <span class="tag">Theme: Pink × Black • Streamlit Demo</span>
      <p style="margin-top:10px;"><b>Nama:</b> IRAWA(N)A JUWITA &nbsp; • &nbsp; <b>NIM:</b> 202210370311446</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Pengaturan")
model_name = st.sidebar.selectbox("Pilih Model", list(MODEL_ZOO.keys()))
uploaded = st.sidebar.file_uploader("Upload gambar bunga", type=["jpg", "jpeg", "png"])

st.sidebar.divider()
st.sidebar.subheader("🧾 History Prediksi")
if st.sidebar.button("🗑️ Hapus History", width="stretch"):
    st.session_state.history = []
    st.sidebar.success("History berhasil dihapus.")
st.sidebar.caption(f"Jumlah history: {len(st.session_state.history)}")

# =========================
# MAIN
# =========================
colA, colB = st.columns([1, 1.25], gap="large")

# --- INPUT CARD
with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🖼️ Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Preview input yang digunakan model (setelah preprocessing letterbox).</div>', unsafe_allow_html=True)
    st.write("")

    if uploaded is None:
        st.info("Upload gambar dulu ya (jpg/png/jpeg).")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        try:
            pil = Image.open(uploaded)
            x, show_img = preprocess_image(pil)
            st.image(show_img, caption="Preview input (dipakai model)", width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Gagal membaca gambar: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

# --- PREDICTION CARD
with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧠 Prediksi</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Hasil prediksi tampil setelah gambar di-upload.</div>', unsafe_allow_html=True)
    st.write("")

    # Load model
    try:
        model = load_keras_model(MODEL_ZOO[model_name])
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    if uploaded is None:
        st.warning("Belum ada gambar yang di-upload.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    try:
        proba = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = CLASS_NAMES[pred_idx]
        conf = float(proba[pred_idx])
        top3 = topk_from_proba(proba, k=3)

        st.markdown(f"### ✅ Prediksi: **{pred_label}**")
        st.markdown(
            f'<span class="badgePink">Confidence: {conf*100:.2f}% &nbsp; (prob={conf:.4f})</span>',
            unsafe_allow_html=True
        )
        st.progress(min(max(conf, 0.0), 1.0))

        st.write("")
        st.markdown("#### 🔝 Top-3 (angka probabilitas)")
        rows = [
            {"Rank": i + 1, "Class": c, "Prob": f"{p:.4f}", "Percent": f"{p*100:.2f}%"}
            for i, (c, p) in enumerate(top3)
        ]
        st.table(rows)

        st.write("")
        st.markdown("#### 📊 Probabilitas semua kelas")
        st.bar_chart({CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))})

        # Save history (newest on top)
        filename = getattr(uploaded, "name", "uploaded_image")
        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "file": filename,
            "pred": pred_label,
            "prob": round(conf, 4),
            "percent": round(conf * 100, 2),
        })

    except Exception as e:
        st.error(
            "Prediksi gagal. Biasanya karena preprocessing tidak sesuai.\n\n"
            f"Detail error:\n{e}"
        )
    finally:
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# HISTORY TABLE
# =========================
st.write("")
st.markdown("### 🧾 History Prediksi (terbaru di atas)")
if len(st.session_state.history) == 0:
    st.info("Belum ada history. Upload gambar untuk mulai.")
else:
    st.dataframe(st.session_state.history, use_container_width=True)

st.caption("© UAP ML — Flower Classification • Streamlit Demo")
