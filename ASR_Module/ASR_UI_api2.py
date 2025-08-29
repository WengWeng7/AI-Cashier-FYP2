import streamlit as st
import requests
import tempfile
from streamlit_mic_recorder import mic_recorder
import numpy as np
import soundfile as sf
import io
import base64
import time
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

# Create two columns for recorder and status
col1, col2 = st.columns([3, 1])

with col1:
    st.write("Click the microphone button and speak clearly...")
    # Audio recorder
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        key="recorder"
    )

with col2:
    if audio:
        st.success("✅ Audio recorded!")
    else:
        st.info("⏸️ Ready")

# Process recorded audio
if audio:
    try:
        st.write("📢 Recorded Audio:")
        st.audio(audio['bytes'])

        with st.spinner("Processing audio..."):
            audio_file = process_audio(audio['bytes'])
            if audio_file:
                with st.spinner("Transcribing..."):
                    result = transcribe_audio(audio_file)
                    if result:
                        st.markdown("### 📝 Transcription:")
                        st.success(result["transcription"])

                        with st.expander("Debug Info"):
                            st.json(result)

        # 👇 Reset recorder state so button unclicks
        st.session_state["recorder"] = None

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")

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

    # Show recording info
    st.markdown("### Recording Settings")
    st.info("""
    📝 Info:
    - Sample Rate: 16kHz
    - Channels: Mono
    - Format: WebM
    """)