# --- Fix Streamlit permission issue BEFORE importing streamlit ---
import os

# Put Streamlit state/config under /tmp (writeable on Spaces)
os.environ.setdefault("STREAMLIT_HOME", "/tmp/streamlit")
os.environ.setdefault("XDG_STATE_HOME", "/tmp")
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp")
os.makedirs(os.environ["STREAMLIT_HOME"], exist_ok=True)

# Disable telemetry to avoid writes under root paths
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

from pathlib import Path
from datetime import datetime
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Urdu TTS - OpenAI", page_icon="🔊", layout="centered")

st.title("🔊 Urdu Text → Speech (OpenAI)")
st.caption("Type Urdu text and generate natural Urdu speech with OpenAI TTS. Use a custom voice_id if your account has Voice access.")

# ---- Get API key (your secret name is Key_1). Fallback to manual entry. ----
API_KEY = os.getenv("Key_1") or st.secrets.get("Key_1")
with st.sidebar:
    st.header("API")
    if not API_KEY:
        st.warning("Secret `Key_1` not found. Paste your key temporarily (will not be saved to repo).")
        API_KEY = st.text_input("OpenAI API key (sk-...)", type="password")
if not API_KEY:
    st.stop()

client = OpenAI(api_key=API_KEY)

# ---- Sidebar options ----
with st.sidebar:
    st.header("Options")
    builtin_voices = ["alloy", "verse", "aria", "ballad", "cove", "luna", "sage"]
    use_custom = st.checkbox("Use custom OpenAI voice_id", False)
    custom_voice = st.text_input("Custom voice_id", value="", help="Requires access to custom voices") if use_custom else ""
    preset_voice = st.selectbox("Built-in voice", builtin_voices, index=0) if not use_custom else None
    out_name = st.text_input("Output filename (no extension)", "urdu_tts")
    fmt = st.selectbox("Audio format", options=["mp3", "wav"], index=0)

# ---- Input ----
default_text = "یہ ایک سادہ مثال ہے۔ یہاں اپنا متن لکھیں اور آڈیو حاصل کریں۔"
text = st.text_area("Urdu text", value=default_text, height=200, placeholder="یہاں اردو میں ٹیکسٹ لکھیں یا پیسٹ کریں…")

col1, col2 = st.columns(2)
with col1:
    make_audio = st.button("🎙️ Generate", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop("audio_bytes", None)
    st.session_state.pop("ext", None)
    st.experimental_rerun()

# ---- TTS helper ----
def synth_openai(urdu_text: str, voice: str, audio_format: str):
    model = "gpt-4o-mini-tts"
    ext = "mp3" if audio_format == "mp3" else "wav"
    tmp_path = Path(f"/tmp/tts_{datetime.now().strftime('%H%M%S')}.{ext}")
    # Stream to file to avoid loading all in memory
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=urdu_text,
        format=("mp3" if audio_format == "mp3" else "wav"),
    ) as resp:
        resp.stream_to_file(tmp_path)
    data = tmp_path.read_bytes()
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass
    return data, ext

# ---- Generate ----
if make_audio:
    if not text.strip():
        st.warning("براہ کرم اردو متن درج کریں")
    else:
        try:
            voice_to_use = (custom_voice.strip() if use_custom else preset_voice)
            if not voice_to_use:
                st.warning("Please choose a built-in voice or provide a custom voice_id.")
            else:
                st.info("Generating Urdu speech with OpenAI TTS…")
                audio_bytes, ext = synth_openai(text.strip(), voice_to_use, fmt)
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state["ext"] = ext
                st.success("آڈیو تیار ہے")
        except Exception as e:
            st.error(f"کچھ مسئلہ آیا: {e}")

# ---- Preview & Download ----
if "audio_bytes" in st.session_state:
    ext = st.session_state.get("ext", "mp3")
    st.markdown("### ▶️ Preview")
    st.audio(st.session_state["audio_bytes"], format=f"audio/{'mpeg' if ext=='mp3' else 'wav'}")
    fname = f"{(out_name or 'urdu_tts').strip()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
    st.download_button(
        "⬇️ Download",
        data=st.session_state["audio_bytes"],
        file_name=fname,
        mime=("audio/mpeg" if ext == "mp3" else "audio/wav"),
        use_container_width=True,
    )

st.markdown("---")
st.caption("If your secret is named `Key_1`, keep it in Settings → Variables and secrets → Secrets. "
           "This app stores Streamlit state under /tmp to avoid permission issues on Spaces.")
