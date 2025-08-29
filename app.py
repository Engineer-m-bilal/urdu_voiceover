import io
import os
import tempfile
from datetime import datetime

import numpy as np
import soundfile as sf
import streamlit as st
from TTS.api import TTS

st.set_page_config(page_title="Urdu Voice Cloner (XTTS v2)", page_icon="🗣️", layout="centered")

st.title("🗣️ Urdu Text → Your Voice (Voice Cloning)")
st.caption("Upload a short sample of your voice, type Urdu text, and get audio in your voice (XTTS v2, CPU friendly).")

# ----------------------------
# Cache the model so it loads once
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_tts():
    # Multilingual zero-shot cloning, supports Urdu with language='ur'
    return TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

tts = load_tts()

# ----------------------------
# Sidebar options
# ----------------------------
with st.sidebar:
    st.header("Options")
    st.caption("Upload a clean 10–30s clip, no background noise if possible.")
    similarity_boost = st.slider("Similarity boost", 0.0, 1.0, 0.75, 0.05)
    stability = st.slider("Stability", 0.0, 1.0, 0.60, 0.05)
    style = st.slider("Style (expressiveness)", 0.0, 1.0, 0.35, 0.05)
    normalize = st.checkbox("Normalize loudness", True)
    base_name = st.text_input("Output filename (no extension)", "urdu_voice_clone")
    seed = st.number_input("Random seed", value=42, step=1)

# ----------------------------
# Simple helpers (no librosa)
# ----------------------------
def simple_trim_silence(wave: np.ndarray, threshold: float = 1e-4, pad: int = 0) -> np.ndarray:
    """
    Very simple silence trim: finds where absolute amplitude exceeds threshold.
    If nothing exceeds threshold, returns original.
    """
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    idx = np.where(np.abs(wave) > threshold)[0]
    if idx.size == 0:
        return wave
    start = max(int(idx[0]) - pad, 0)
    end = min(int(idx[-1]) + pad, wave.shape[0])
    return wave[start:end]

def normalize_peak(wave: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = np.max(np.abs(wave)) + 1e-9
    return (peak * wave / m).astype(np.float32)

def wav_bytes_from_array(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()

# ----------------------------
# Inputs
# ----------------------------
ref_file = st.file_uploader("Upload your voice sample (wav/mp3/m4a/ogg/flac)", type=["wav", "mp3", "m4a", "ogg", "flac"])
default_text = "یہ میری آواز کی مثال ہے۔ آپ یہاں اپنا متن لکھیں اور آڈیو حاصل کریں۔"
text = st.text_area("Urdu text", value=default_text, height=180, placeholder="یہاں اردو میں ٹیکسٹ لکھیں یا پیسٹ کریں…")

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("🎙️ Generate", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop("audio_bytes", None)
    st.experimental_rerun()

# ----------------------------
# Run synthesis
# ----------------------------
if run_btn:
    if not text.strip():
        st.warning("براہ کرم اردو متن درج کریں۔")
    elif ref_file is None:
        st.warning("براہ کرم اپنی آواز کی آڈیو فائل اپلوڈ کریں۔")
    else:
        try:
            # Save uploaded file to a temp path (XTTS can accept various formats via soundfile/ffmpeg backend)
            tmp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ref_file.name.split('.')[-1]}")
            tmp_ref.write(ref_file.read())
            tmp_ref.flush()
            tmp_ref.close()

            # Optional: quick silence trim to reduce leading/trailing gaps
            try:
                y_ref, sr_ref = sf.read(tmp_ref.name, dtype="float32", always_2d=False)
                y_ref = simple_trim_silence(y_ref)
                sf.write(tmp_ref.name, y_ref, sr_ref)  # overwrite trimmed
            except Exception:
                # If reading/trim fails, keep original file
                pass

            st.info("Cloning voice and synthesizing Urdu… (CPU can take a bit on first run)")

            out_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

            # Generate audio
            tts.tts_to_file(
                text=text.strip(),
                file_path=out_wav_path,
                speaker_wav=tmp_ref.name,
                language="ur",
                # Common conditioning knobs
                speaker_similarity=float(similarity_boost),
                stability=float(stability),
                style=float(style),
                split_sentences=True,
                seed=int(seed),
            )

            # Load, optional normalize, then serve
            y, sr = sf.read(out_wav_path, dtype="float32", always_2d=False)
            if normalize:
                y = normalize_peak(y)

            audio_bytes = wav_bytes_from_array(y, sr)
            st.session_state["audio_bytes"] = audio_bytes

            # Cleanup temp files
            try:
                os.remove(tmp_ref.name)
                os.remove(out_wav_path)
            except Exception:
                pass

            st.success("آڈیو تیار ہے۔")

        except Exception as e:
            st.error(f"کچھ مسئلہ آیا: {e}")

# ----------------------------
# Preview & download
# ----------------------------
if "audio_bytes" in st.session_state:
    st.markdown("### ▶️ Preview")
    st.audio(st.session_state["audio_bytes"], format="audio/wav")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{(base_name or 'urdu_voice_clone').strip()}_{ts}.wav"
    st.download_button("⬇️ Download WAV", data=st.session_state["audio_bytes"], file_name=fname, mime="audio/wav", use_container_width=True)

st.markdown("---")
st.caption(
    "Tips: Use a clear 10–30 second reference with low noise. If cloning feels off, try a different sample, "
    "raise Similarity slightly, or lower Stability a little."
)
