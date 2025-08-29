import io
from datetime import datetime
from gtts import gTTS
import streamlit as st

st.set_page_config(page_title="Urdu Voice Over", page_icon="🔊", layout="centered")

st.title("🔊 Urdu Text → Voice")
st.caption("Type or paste Urdu text and get an MP3 voice over. Works great on Hugging Face Spaces.")

# Sidebar options
with st.sidebar:
    st.header("Options")
    slow = st.checkbox("Slow reading", False)
    file_basename = st.text_input("Output filename (without extension)", "urdu_voice")
    st.markdown("---")
    st.write("Tips")
    st.caption("1) Paste clean Urdu text\n2) Use Slow reading if the voice feels fast\n3) Download the MP3 for editing")

# Main input
default_sample = "یہ ایک سادہ مثال ہے، آپ یہاں اپنا متن لکھیں اور آڈیو حاصل کریں۔"
text = st.text_area(
    "Urdu text",
    value=default_sample,
    height=200,
    placeholder="یہاں اردو میں ٹیکسٹ لکھیں یا پیسٹ کریں…"
)

def tts_to_bytes(urdu_text: str, slow_read: bool = False) -> bytes:
    # gTTS automatically chunks long text
    tts = gTTS(text=urdu_text, lang="ur", slow=slow_read)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

col1, col2 = st.columns([1,1])
with col1:
    make_audio = st.button("🎙️ Generate Voice", use_container_width=True)
with col2:
    clear = st.button("🧹 Clear", use_container_width=True)

if clear:
    st.session_state.pop("audio_bytes", None)
    st.session_state.pop("last_text", None)
    st.experimental_rerun()

if make_audio:
    if not text.strip():
        st.warning("براہ کرم آڈیو بنانے کے لیے اردو متن درج کریں")
    else:
        try:
            audio_bytes = tts_to_bytes(text.strip(), slow_read=slow)
            st.session_state["audio_bytes"] = audio_bytes
            st.session_state["last_text"] = text.strip()
            st.success("آڈیو تیار ہے")
        except Exception as e:
            st.error(f"کچھ مسئلہ آیا: {e}")

# Playback and download
if "audio_bytes" in st.session_state:
    st.markdown("### ▶️ Preview")
    st.audio(st.session_state["audio_bytes"], format="audio/mp3")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = file_basename.strip() or "urdu_voice"
    fname = f"{base}_{timestamp}.mp3"

    st.download_button(
        label="⬇️ Download MP3",
        data=st.session_state["audio_bytes"],
        file_name=fname,
        mime="audio/mpeg",
        use_container_width=True
    )

st.markdown("---")
st.caption(
    "Note: This app uses gTTS for Urdu speech synthesis. If your Space is very busy or internet is restricted, synthesis can fail. Try again after a short while."
)
