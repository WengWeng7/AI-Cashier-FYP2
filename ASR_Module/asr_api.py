from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io
from pydub import AudioSegment
import soundfile as sf
import torch
import os
import csv
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from transformers.models.whisper import tokenization_whisper
import time

app = FastAPI(title="ASR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = None

os.makedirs("wav", exist_ok=True)
os.makedirs("text", exist_ok=True)

def load_model(model_name="mesolitica/malaysian-whisper-small-v2", return_timestamps=False):
    global pipe
    if pipe is None:
        print("🔄 Loading ASR model...")
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps=return_timestamps,
            device=0 if torch.cuda.is_available() else -1,
        )
        print("✅ Model loaded.")
    return pipe

@app.on_event("startup")
async def startup_event():
    load_model(return_timestamps=False)

# Transcription metadata helper
def get_next_index(session_id: str) -> int:
    """Get next index for the given session_id by scanning metadata.csv"""
    if not os.path.exists("metadata.csv"):
        return 1

    last_index = 0
    with open("metadata.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        for row in reader:
            if row and row[0] == session_id:
                # row[1] looks like sessionid_3.wav → extract "3"
                try:
                    idx = int(row[1].split("_")[-1].replace(".wav", ""))
                    if idx > last_index:
                        last_index = idx
                except:
                    pass
    return last_index + 1


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), session_id: str = Form(...)):
    audio_bytes = await file.read()

    # Convert WebM → WAV (always safe regardless of browser)
    audio_io = io.BytesIO(audio_bytes)
    audio = AudioSegment.from_file(audio_io, format="webm")

    # Resample → 16k mono WAV
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    audio_np, samplerate = sf.read(wav_io)

    if pipe is None:
        load_model()

    start_time = time.time()
    # 🔑 Force "transcribe" every time
    result = pipe(
        {"array": audio_np, "sampling_rate": samplerate},
        generate_kwargs={"task": "transcribe"}
    )
    end_time = time.time()
    transcription = result["text"]
    print(f"[DEBUG] Transcribed text: {transcription}")
    print(f"[DEBUG] Transcription time: {end_time - start_time:.2f} seconds")
    
    # Determine sequence number for this session
    index = get_next_index(session_id)

    wav_filename = f"{session_id}_{index}.wav"
    txt_filename = f"{session_id}_{index}.txt"

    # Save WAV file
    wav_path = os.path.join("wav", wav_filename)
    audio.export(wav_path, format="wav")

    # Save TXT file
    txt_path = os.path.join("text", txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    # Update metadata.csv (append mode)
    with open("metadata.csv", "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        writer.writerow([session_id, wav_filename, transcription])
        
    return JSONResponse({"transcription": transcription})

@app.get("/health")
def health_check():
    return {"status": "ok"}
