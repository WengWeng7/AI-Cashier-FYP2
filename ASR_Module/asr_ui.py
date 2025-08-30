import streamlit as st
import requests
import tempfile
from streamlit_realtime_audio_recorder import audio_recorder
import io
import base64
from pydub import AudioSegment

# Page config
st.set_page_config(page_title="ASR Demo", page_icon="🎤")
st.title("🎤 Speech Recognition Demo")

API_URL = "http://127.0.0.1:8000/transcribe"

def process_audio(audio_bytes):
    """Convert WebM audio bytes to WAV and save to temp file"""
    try:
        # Decode webm bytes with pydub
        audio_io = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_io, format="webm")

        # Ensure mono + 16kHz for ASR
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Save as temporary WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio.export(tmp.name, format="wav")
            return tmp.name

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def transcribe_audio(file_path):
    """Send audio file to API for transcription"""
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(API_URL, files={'file': f})
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

# Main interface
st.markdown("### 🎙️ Record your voice")
st.write("Click the microphone button and speak clearly...")

# Audio recorder with real-time capabilities
result = audio_recorder(
    interval=50,  # Update interval in milliseconds
    threshold=-60,  # Audio threshold for voice detection
    silenceTimeout=200  # Timeout in milliseconds
)

# Process recorded audio
if result:
    if result.get('status') == 'stopped':
        audio_data = result.get('audioData')
        if audio_data:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Display audio player
            st.write("📢 Recorded Audio:")
            st.audio(io.BytesIO(audio_bytes), format="audio/webm")
            
            # Process and transcribe
            with st.spinner("Processing audio..."):
                audio_file = process_audio(audio_bytes)
                if audio_file:
                    # Transcribe
                    with st.spinner("Transcribing..."):
                        result = transcribe_audio(audio_file)
                        if result:
                            st.markdown("### 📝 Transcription:")
                            st.success(result["transcription"])
                            
                            # Debug info
                            with st.expander("Debug Info"):
                                st.json(result)
        else:
            st.warning("No audio data was recorded")
    elif result.get('error'):
        st.error(f"Error: {result.get('error')}")

# Health check and status
with st.sidebar:
    st.markdown("### System Status")
    try:
        health = requests.get(f"http://127.0.0.1:8000/health")
        if health.status_code == 200:
            st.success("✅ API is online")
        else:
            st.error("❌ API is offline")
    except:
        st.error("❌ Cannot connect to API")