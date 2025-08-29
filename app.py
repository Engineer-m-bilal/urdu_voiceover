import io
import os
import tempfile
from datetime import datetime

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from TTS.api import TTS

st.set_page_config(page_title="Urdu Voice Cloner", page_icon="🗣️", layout="centered")

st.title("🗣️ Urdu Text → Your Voice (Voice Cloning)")
st.caption("Upload a short sample of your voice, type Urdu text, and get audio in your voice.")

# ----------------------------
# Caching the model to avoid reloading
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_tts():
    # XTTS v2 supports multilingual zero-shot cloning, including Urdu (code: 'ur')
    # Model will download on first run and then be cached by the Space
    return TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

tts = load_tts()

# ----------------------------
# Sidebar: options
# ----------------------------
with st.sidebar:
    st.header("Options")
    st.markdown("**Reference voice**")
    st.caption("Upload a clean 10–30 second sample with minimal noise.")
    # XTTS controls
    similarity_boost = st.slider("Similarity boost", 0.0, 1.0, 0.75, 0.05)
    stability = st.slider("Stability", 0.0, 1.0, 0.6, 0.05)
    style = st.slider("Style (expressiveness)", 0.0, 1.0, 0.35, 0.05)
    seed = st.number_input("Random seed (for reproducibility)", value=42, step=1)

    st.markdown("---")
    st.markdown("**Post-processing**")
    rate = st.slider("Speaking rate (time-stretch)", 0.75, 1.25, 1.00, 0.01)
    normalize = st.checkbox("Normalize loudness", True)

    st.markdown("---")
    base_name = st.text_input("Output filename (no extension)", "urdu_voice_clone")

# ----------------------------
# Inputs
# ----------------------------
ref_file = st.file_uploader(
    "Upload your voice sample (wav/mp3/m4a)", 
    type=["wav", "mp3", "m4a", "ogg", "flac"]
)

default_text = "یہ میری آواز کی مثال ہے۔ آپ یہاں اپنا متن لکھیں اور آڈیو حاصل کریں۔"
text = st.text_area(
    "Urdu text",
    value=default_text,
    height=180,
    placeholder="یہاں اردو میں ٹیکسٹ لکھیں یا پیسٹ کریں…"
)

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("🎙️ Generate", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop("audio_bytes", None)
    st.session_state.pop("preview_sr", None)
    st.experimental_rerun()

# ----------------------------
# Helpers
# ----------------------------
def load_and_standardize(audio_file, target_sr=16000):
    """Load user audio, convert to mono 16 kHz WAV bytes and return temp path."""
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    if len(y) < target_sr * 3:
        st.warning("Voice sample is very short. Try at least 5–10 seconds for better cloning.")
    y_res = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # Light trim to remove leading/trailing silence
    yt, _ = librosa.effects.trim(y_res, top_db=30)
    if yt.size < target_sr:  # ensure at least 1s remains
        yt = y_res
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_wav.name, yt, target_sr)
    return tmp_wav.name

def postprocess_rate_and_norm(wav, sr, rate_factor=1.0, do_norm=True):
    """Time-stretch and normalize loudness."""
    y = wav.astype(np.float32)
    if rate_factor != 1.0:
        # librosa requires strictly positive values
        y = librosa.effects.time_stretch(y, rate_factor)
    if do_norm:
        peak = np.max(np.abs(y)) + 1e-9
        y = 0.98 * (y / peak)
    return y

def wav_bytes_from_array(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()

# ----------------------------
# Run
# ----------------------------
if run_btn:
    if not text.strip():
        st.warning("براہ کرم اردو متن درج کریں۔")
    elif ref_file is None:
        st.warning("براہ کرم اپنی آواز کی آڈیو فائل اپلوڈ کریں۔")
    else:
        try:
            st.info("Preparing reference voice…")
            ref_path = load_and_standardize(ref_file)

            st.info("Cloning voice and synthesizing Urdu…")
            # Generate to a temporary file first
            out_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

            # Coqui XTTS generation
            # Extra params passed via "speaker_wav" and "language"
            # Controls: "speaker_similarity", "style", "temperature", "length_scale" etc. are model dependent.
            tts.tts_to_file(
                text=text.strip(),
                file_path=out_wav_path,
                speaker_wav=ref_path,
                language="ur",
                # Extra inference kwargs routed to the model (supported by XTTS v2)
                # See: https://github.com/coqui-ai/TTS
                # Using similarity/stability/style through speaker conditioning
                # Some builds accept these as speaker_cfg; we forward common names:
                split_sentences=True,
                speed=1.0,  # base speed, we will also post-process rate
                speaker_similarity=similarity_boost,
                stability=stability,
                style_wav=None,
                style=style,
                seed=int(seed)
            )

            # Read back and post-process
            y, sr = sf.read(out_wav_path, dtype="float32")
            y = postprocess_rate_and_norm(y, sr, rate_factor=rate, do_norm=normalize)
            audio_bytes = wav_bytes_from_array(y, sr)

            # Stash in session for preview and download
            st.session_state["audio_bytes"] = audio_bytes
            st.session_state["preview_sr"] = sr

            # Clean temp files
            try:
                os.remove(ref_path)
                os.remove(out_wav_path)
            except Exception:
                pass

            st.success("آڈیو تیار ہے۔ نیچے سنیں یا ڈاؤن لوڈ کریں۔")

        except Exception as e:
            st.error(f"کچھ مسئلہ آیا: {e}")

# ----------------------------
# Preview and download
# ----------------------------
if "audio_bytes" in st.session_state:
    st.markdown("### ▶️ Preview")
    st.audio(st.session_state["audio_bytes"], format="audio/wav")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (base_name or "urdu_voice_clone").strip()
    fname = f"{base}_{ts}.wav"

    st.download_button(
        "⬇️ Download WAV",
        data=st.session_state["audio_bytes"],
        file_name=fname,
        mime="audio/wav",
        use_container_width=True
    )

st.markdown("---")
st.caption(
    "Tips: Use a clear 10–30 second reference with low noise. Speak naturally. "
    "If cloning feels off, try a different sample, raise Similarity, or lower Stability a little."
)
