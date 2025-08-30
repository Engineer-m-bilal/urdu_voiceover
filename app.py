import io
import os
from pathlib import Path
from datetime import datetime
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Urdu TTS - OpenAI", page_icon="🔊", layout="centered")

st.title("🔊 Urdu Text → Speech (OpenAI)")
st.caption("Type Urdu text and generate natural Urdu speech with OpenAI TTS. If you have a custom voice ID, you can use it.")

# ---- API key from Hugging Face Secret ----
API_KEY = os.getenv("Key_1") or st.secrets.get("Key_1")
if not API_KEY:
    st.error("Missing Key_1. Go to Settings → Secrets in your Space and add it.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# ---- Sidebar options ----
with st.sidebar:
    st.header("Options")
    voices = ["alloy", "verse", "aria", "ballad", "cove", "luna", "sage"]
    use_custom = st.checkbox("Use custom OpenAI voice ID", False)
    custom_voice = ""
    if use_custom:
        custom_voice = st.text_input("Custom voice_id", value="", help="Only works if your account has custom voices enabled")
    else:
        preset = st.selectbox("Built-in voice", options=voices, index=0)
    out_name = st.text_input("Output filename (no extension)", "urdu_tts")
    fmt = st.selectbox("Audio format", options=["mp3", "wav"], index=0)

# ---- Main input ----
default_text = "یہ ایک سادہ مثال ہے۔ یہاں اپنا متن لکھیں اور آڈیو حاصل کریں۔"
text = st.text_area("Urdu text", value=default_text, height=200, placeholder="یہاں اردو میں ٹیکسٹ لکھیں یا پیسٹ کریں…")

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("🎙️ Generate", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop("audio_bytes", None)
    st.experimental_rerun()

# ---- Helper function ----
def tts_openai(urdu_text: str, voice: str, audio_format: str) -> bytes:
    model = "gpt-4o-mini-tts"
    ext = "mp3" if audio_format == "mp3" else "wav"
    tmp_path = Path(f"/tmp/tts_{datetime.now().strftime('%H%M%S')}.{ext}")
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=urdu_text,
        format=audio_format
    ) as resp:
        resp.stream_to_file(tmp_path)
    data = tmp_path.read_bytes()
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass
    return data, ext

# ---- Generate speech ----
if run_btn:
    if not text.strip():
        st.warning("براہ کرم اردو متن درج کریں")
    else:
        try:
            voice_to_use = custom_voice.strip() if use_custom and custom_voice.strip() else (preset if not use_custom else None)
            if not voice_to_use:
                st.warning("Custom voice ID is empty. Either uncheck custom voice or provide a valid voice_id.")
            else:
                st.info("Generating Urdu speech with OpenAI TTS…")
                audio_bytes, ext = tts_openai(text.strip(), voice_to_use, fmt)
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state["ext"] = ext
                st.success("آڈیو تیار ہے")
        except Exception as e:
            st.error(f"کچھ مسئلہ آیا: {e}")

# ---- Preview & download ----
if "audio_bytes" in st.session_state:
    ext = st.session_state.get("ext", "mp3")
    st.markdown("### ▶️ Preview")
    st.audio(st.session_state["audio_bytes"], format=f"audio/{'mpeg' if ext=='mp3' else 'wav'}")
    fname = f"{(out_name or 'urdu_tts').strip()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
    st.download_button(
        "⬇️ Download",
        data=st.session_state["audio_bytes"],
        file_name=fname,
        mime="audio/mpeg" if ext == "mp3" else "audio/wav",
        use_container_width=True
    )

st.markdown("---")
st.caption(
    "Notes: Built-in voices are not your personal voice. For your own cloned voice, you need access to custom OpenAI Voice IDs. "
    "Urdu is supported by gpt-4o-mini-tts."
)
